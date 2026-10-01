import threading
from types import SimpleNamespace
from unittest.mock import Mock

import openai
import pytest

from src.visual_pilot import config, db, llm, pmc, queue_state, select_articles


def test_provider_error_payload_is_retryable_and_preserves_message(monkeypatch):
    client = llm.LLMClient(max_retries=1)
    failure = SimpleNamespace(choices=None,error={'code':503,'message':'upstream unavailable'})
    success = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{}'))])
    create = Mock(side_effect=[failure,success])
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr(llm.time,'sleep',lambda seconds: None)
    response,_ = client._send_once('space-bunny-test','system','user',{'type':'object'},[])
    assert response is success
    assert create.call_count == 2
    client.max_retries = 0
    create.side_effect = [failure]
    with pytest.raises(openai.InternalServerError,match='upstream unavailable') as error:
        client._send_once('space-bunny-test','system','user',{'type':'object'},[])
    assert error.value.response.request.url.path.endswith('/chat/completions')


def test_initialization_constructs_one_shared_client(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    client=llm.LLMClient()
    client.api_key='test'
    factory=Mock(return_value=object())
    monkeypatch.setattr(llm.openai,'OpenAI',factory)
    with ThreadPoolExecutor(max_workers=16) as pool:
        values=list(pool.map(lambda _: client._openai(),range(64)))
    assert factory.call_count == 1
    assert all(v is values[0] for v in values)


def test_permanent_provider_error_does_not_retry(monkeypatch):
    client=llm.LLMClient(max_retries=3)
    create=Mock(return_value=SimpleNamespace(choices=[],error={'code':401,'message':'invalid credential'}))
    client._client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    with pytest.raises(openai.AuthenticationError,match='invalid credential'):
        client._send_once('space-bunny-test','system','user',{'type':'object'},[])
    assert create.call_count == 1


def _database(tmp_path):
    path=tmp_path/'queue.sqlite'
    conn=db.connect(path)
    db.init_db(conn)
    for pmcid in ('A','B','C'):
        conn.execute("INSERT INTO articles(pmcid,status,error) VALUES(?,'license_ok','old error')",(pmcid,))
    conn.commit()
    return conn,path


def test_completion_order_is_checkpointed_before_interruption(tmp_path):
    conn,path=_database(tmp_path)
    verdict={'decision':'relevant','is_narrative_review':True,'primary_disease_keys':['as'],'reason':'confirmed'}

    def results(requests):
        yield llm.BatchResult(index=1,parsed=verdict)
        other=db.connect(path)
        assert other.execute("SELECT status,error FROM articles WHERE pmcid='B'").fetchone()[:] == ('relevant',None)
        other.close()
        raise KeyboardInterrupt()

    client=SimpleNamespace(_lock=threading.Lock(),iter_many=results)
    for pmcid in ('A','B'):
        queue_state.record(conn,pmcid,'relevance','running','submitted')
    conn.commit()
    with pytest.raises(KeyboardInterrupt):
        select_articles.process_relevance(conn,client,[],['A','B'])
    assert conn.execute("SELECT status,reason FROM article_queue_state WHERE pmcid='A'").fetchone()[:] == ('deferred','interruption')
    assert conn.execute("SELECT status FROM articles WHERE pmcid='B'").fetchone()[0]=='relevant'


def test_completed_result_delivered_before_request_feed_error(monkeypatch):
    client=llm.LLMClient(concurrency=1)
    monkeypatch.setattr(client,'call_json',lambda **kw: ({'ok':True},{}))
    def feed():
        yield {}
        raise RuntimeError('producer failed')
    results=client.iter_many(feed())
    assert next(results).parsed == {'ok':True}
    with pytest.raises(RuntimeError,match='producer failed'):
        next(results)


def test_errors_and_budget_are_durable_and_remain_resumable(tmp_path):
    conn,_=_database(tmp_path)
    client=SimpleNamespace(_lock=threading.Lock(),iter_many=lambda _: iter([
        llm.BatchResult(index=1,error=llm.BudgetExceeded('budget exhausted')),
        llm.BatchResult(index=0,error=RuntimeError('provider unavailable'))]))
    counts=select_articles.process_relevance(conn,client,[],['A','B'])
    assert counts == {'completed':0,'errors':1,'budget':1}
    assert conn.execute("SELECT status FROM articles WHERE pmcid='A'").fetchone()[0]=='license_ok'
    assert conn.execute("SELECT reason FROM article_queue_state WHERE pmcid='B'").fetchone()[0]=='budget'


def test_absent_license_field_uses_fallback_but_nc_does_not(monkeypatch):
    monkeypatch.setattr(pmc,'get_licenses_epmc',lambda ids: {
        'A':pmc.LicenseInfo(code='none',url=None,oa_subset='oa',raw=None,source='epmc'),
        'B':pmc.LicenseInfo(code='cc-by-nc',url=None,oa_subset='oa',raw='CC BY-NC',source='epmc')})
    fallback=Mock(return_value=('A',{'status':'license_ok','license_code':'cc-by','license_raw':'CC BY'}))
    monkeypatch.setattr(select_articles,'join_license',fallback)
    outcomes=dict(select_articles.join_licenses(['A','B']))
    assert fallback.call_args.args == ('A',)
    assert outcomes['A']['fallback_reason']=='missing_epmc_license'
    assert outcomes['B']['status']=='license_rejected'
    assert outcomes['B']['license_source']=='epmc'


def test_license_access_failure_is_retryable(monkeypatch,tmp_path):
    conn,_=_database(tmp_path)
    monkeypatch.setattr(pmc,'get_license',Mock(side_effect=pmc.PmcError('temporary outage')))
    pmcid,outcome=select_articles.join_license('A')
    assert select_articles.apply_license(conn,pmcid,outcome)=='candidate'
    assert conn.execute("SELECT reason FROM article_queue_state WHERE pmcid='A'").fetchone()[0]=='access_error'


def test_manifestation_deficit_overrides_disease_quota(tmp_path,monkeypatch):
    conn,_=_database(tmp_path)
    conn.execute("INSERT INTO diseases(disease_key,name) VALUES('as','AS')")
    conn.execute("INSERT INTO findings_vocab(finding_key,disease_keys_json,label,category,approved) VALUES('uveitis','[\"as\"]','Uveitis','eye',1)")
    for pmcid in ('A','B','C'):
        conn.execute(
            "INSERT INTO manifestation_candidates"
            "(disease_key,finding_key,pmcid,provenance_status,provenance_disease_key) "
            "VALUES('as','uveitis',?,'explicit','as')", (pmcid,))
    conn.execute("UPDATE articles SET status='candidate' WHERE pmcid='C'")
    monkeypatch.setattr(config,'VP_MANIFESTATION_QUOTA',3)
    assert select_articles.needs_manifestation_licenses(conn,'as')
    monkeypatch.setattr(config,'VP_MANIFESTATION_QUOTA',2)
    assert not select_articles.needs_manifestation_licenses(conn,'as')
    conn.execute("UPDATE articles SET status='irrelevant' WHERE pmcid='B'")
    assert select_articles.needs_manifestation_licenses(conn,'as')
    conn.execute("UPDATE articles SET status='relevant',primary_disease_keys_json='[\"ra\"]' WHERE pmcid='B'")
    assert select_articles.needs_manifestation_licenses(conn,'as')
    conn.execute("UPDATE articles SET primary_disease_keys_json='[\"as\"]' WHERE pmcid='B'")
    assert not select_articles.needs_manifestation_licenses(conn,'as')
