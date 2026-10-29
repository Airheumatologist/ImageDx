import json,re,unicodedata,collections,difflib
from pathlib import Path
BASE=Path(__file__).parent
HEADINGS=json.loads((BASE/'all_headings.json').read_text()); HD={x['uid']:x for x in HEADINGS}; OLD=json.loads((BASE/'candidates.json').read_text()); EXIST=json.loads((BASE/'existing_catalog.json').read_text());EIDS={x['topic_id'] for x in EXIST}
def norm(s):return re.sub(r'[^a-z0-9]','',unicodedata.normalize('NFKD',s).encode('ascii','ignore').decode().lower())
def slug(s):return re.sub(r'[^a-z0-9]+','_',unicodedata.normalize('NFKD',s).encode('ascii','ignore').decode().lower()).strip('_')
ID_UID={re.search(r'Topic ID:\s*([^\n]+)',x['body']).group(1).strip():x['uid'] for x in HEADINGS if 'Topic ID:' in x['body']}
files=['review_breast_urology.json','review_cardiology.json','review_gastroenterology.json','review_endo_heme.json','review_ent_neph_immuno.json','review_neuro_rheum.json','review_pediatrics.json','review_orthopedics.json','review_pulmonology.json']
rows=[]
for fn in files:
 if not (BASE/fn).exists():continue
 for i,x in enumerate(json.loads((BASE/fn).read_text())):
  x=dict(x);x['review_file']=fn
  if not x.get('uid') and x.get('index'):x['uid']=ID_UID.get(OLD[x['index']-1]['topic_id'])
  x['section']=x.get('section') or HD.get(x.get('uid'),{}).get('section','Unassigned')
  if x.get('uid') in HD:x['source_heading']=HD[x['uid']]['name']
  else:x['source_heading']=x.get('source_heading',x['canonical_name'])
  x['decision_id']=f'{fn.removeprefix("review_").removesuffix(".json")}_{i+1:03}'
  x.setdefault('existing_parent_ids',[]);x.setdefault('source_urls',[])
  rows.append(x)
# A source heading missed by the Topic-ID-only parser.
rows.append(dict(uid='D0965',section='Gastroenterology',source_heading=HD['D0965']['name'],decision_id='integration_001',proposed_id='type_2_autoimmune_pancreatitis',canonical_name='Type 2 autoimmune pancreatitis',action='DEFER',priority='P2',visual_anchor='Focal or diffuse pancreatic enlargement overlaps type 1 autoimmune pancreatitis and pancreatic carcinoma.',rationale='Distinct IgG4 unrelated disease, but no reliable separate clinical or radiologic discriminator is supplied; do not equate it with IgG4 related disease.',existing_parent_ids=[],source_urls=[]))
# Cross-specialty synonyms and deliberate single-topic homes.
GROUPS=[
('Fibromuscular dysplasia','Cardiology',['Renovascular fibromuscular dysplasia']),
('Middle aortic syndrome','Cardiology',['Mid-aortic syndrome']),
('Nutcracker syndrome','Nephrology',[]),
('Renal osteodystrophy and secondary hyperparathyroidism','Nephrology',['Renal osteodystrophy (CKD mineral and bone disorder)']),
('Encapsulating peritoneal sclerosis','Gastroenterology',['Sclerosing Encapsulating Peritonitis']),
('Persistent Mullerian duct syndrome','Urology',['Persistent Müllerian duct syndrome']),
('Heterotaxy syndrome','Cardiology',['Heterotaxy Syndrome (Situs Ambiguus)']),
('Intestinal duplication cyst','Gastroenterology',['Enteric Duplication Cyst']),
('Germinal matrix and intraventricular hemorrhage of prematurity','Pediatrics',['Germinal Matrix–Intraventricular Hemorrhage']),
('Periventricular leukomalacia','Pediatrics',[]),('Kernicterus','Pediatrics',[]),
('Airway foreign body aspiration','Pulmonology',['Airway foreign body','Airway Foreign-Body Aspiration','Foreign body aspiration']),
('Physical child abuse','Pediatrics',['Nonaccidental skeletal trauma','Physical Child Abuse: Skeletal and Intracranial Imaging']),
('Hair thread tourniquet syndrome','Pediatrics',['Hair-Thread Tourniquet Syndrome','Penile hair-tourniquet syndrome']),
('Ependymoma','Neurology',['Posterior Fossa Ependymoma']),
('Choroid plexus tumors','Neurology',['Choroid Plexus Papilloma','Choroid Plexus Carcinoma','Choroid Plexus Tumors','Choroid Plexus Tumors (Papilloma and Carcinoma)']),
('Hydranencephaly','Pediatrics',[]),
('Perinatal arterial ischemic stroke','Pediatrics',[]),
('Acute ischemic stroke','Neurology',[]),
('Neuralgic amyotrophy','Neurology',[]),
('Neuropathic arthropathy','Orthopedics',['Neuropathic osteoarthropathy (Charcot)','Neuropathic arthropathy (Charcot joint)']),
('Secondary hypertrophic osteoarthropathy','Rheumatology',[]),
('Transient osteoporosis of the hip','Orthopedics',[]),
('Spinal epidural abscess','Neurology',[]),
('Pallister Hall syndrome','Pediatrics',[]),
('Wiedemann Rautenstrauch syndrome','Pediatrics',[]),
('Jansen metaphyseal chondrodysplasia','Orthopedics',[]),
('Frontometaphyseal dysplasia','Orthopedics',[]),
('Mesenchymal hamartoma of the liver','Pediatrics',[]),
('Neonatal mastitis','Pediatrics',[]),
('Neuroendocrine cell hyperplasia of infancy','Pulmonology',[]),
('Pleuropulmonary blastoma','Pediatrics',[]),
('Pulmonary artery sling','Cardiology',[]),
('Tracheobronchomalacia','Pulmonology',['Tracheomalacia']),
('Idiopathic subglottic stenosis','Otolaryngology',[]),
('WHIM syndrome','Immunology',[]),
('Severe congenital neutropenia','Immunology',['Severe congenital neutropenia (Kostmann)']),
('Hepatic sinusoidal obstruction syndrome','Gastroenterology',[]),
('Alveolar capillary dysplasia with misalignment of pulmonary veins','Pulmonology',['Alveolar capillary dysplasia']),
('MALT lymphoma','Hematology',['Gastric MALT Lymphoma','Extranodal marginal zone (MALT) lymphoma']),
('Splanchnic venous thrombosis','Gastroenterology',['Splanchnic vein thrombosis','Portal Vein Thrombosis and Pylephlebitis']),
('HNF1B related kidney disease','Nephrology',['HNF1B-related kidney disease (RCAD)']),
('Spondyloenchondrodysplasia with immune dysregulation','Immunology',['Spondyloenchondrodysplasia with immune dysregulation (SPENCD)']),
('Autoimmune lymphoproliferative syndrome','Immunology',['Autoimmune lymphoproliferative syndrome (ALPS)']),
('L1 syndrome','Pediatrics',['L1 Syndrome (X-Linked Hydrocephalus)']),
('Microcephalic osteodysplastic primordial dwarfism type II','Pediatrics',['MOPD Type II']),
('Matthew Wood syndrome','Pediatrics',['Matthew-Wood Syndrome (PDAC)']),
('Atelosteogenesis','Orthopedics',['Boomerang Dysplasia']),
('COPA syndrome','Immunology',[]),
('Superior semicircular canal dehiscence syndrome','Otolaryngology',['Superior semicircular canal dehiscence']),
('Neuroendocrine cell hyperplasia of infancy','Pulmonology',['Neuroendocrine Cell Hyperplasia of Infancy (NEHI)']),
('Asbestos related nonmalignant thoracic disease','Pulmonology',['Asbestosis','Benign Asbestos Pleural Disease']),
('Inert metal pneumoconiosis','Pulmonology',['Siderosis','Stannosis','Baritosis']),
('Pulmonary agenesis and aplasia','Pulmonology',['Pulmonary Agenesis','Pulmonary Aplasia']),
('Genetic surfactant dysfunction','Pulmonology',['Surfactant Protein B Deficiency','SFTPC-Related Interstitial Lung Disease','ABCA3-Related Surfactant Deficiency','NKX2-1-Related Lung Disease','ABCA3-Related Surfactant Dysfunction','Surfactant Protein C Dysfunction']),
('Alveolar capillary dysplasia with misalignment of pulmonary veins','Pulmonology',['Alveolar Capillary Dysplasia with Misalignment of Pulmonary Veins (ACD/MPV)']),
]
MAP={norm(n):(title,home) for title,home,aliases in GROUPS for n in [title,*aliases]}
for x in rows:
 if norm(x['canonical_name']) in MAP:
  title,home=MAP[norm(x['canonical_name'])];x['final_name']=title;x['home_specialty']=home
 else:x['final_name']=x['canonical_name'];x['home_specialty']=x['section']
 x['final_id']=slug(x['final_name'])
# Additional independent editorial decisions, preserved in the audit.
for x in rows:
 uid=x.get('uid');name=norm(x['canonical_name'])
 def override(action,reason,parents=None):
  x['action']=action;x['rationale']=reason
  if parents is not None:x['existing_parent_ids']=parents
 if uid in ['D0548','D0550']:
  override('MERGE_EXISTING','Steroidogenic subtype of congenital adrenal hyperplasia. Broaden the existing CAH page beyond 21 hydroxylase deficiency and retain cause specific phenotype labels.',['congenital_adrenal_hyperplasia'])
 if uid=='D0563':override('DEFER','Distinct sex development disorder, but the proposed genital phenotype overlaps other steroidogenesis defects and supplies no separate radiologic discriminator.',[])
 if uid=='D0988':override('DEFER','A distinct intestinal lymphoma defined largely by histology and immunophenotype; the proposed ulcer and wall thickening images do not reliably distinguish it.',[])
 if uid=='D1005':override('DEFER','The supplied bleeding and intracranial hemorrhage appearances are nonspecific; diagnosis depends on coagulation testing rather than an independent visual phenotype.',[])
 if uid=='D1042':override('DEFER','Distinct cutaneous lymphoma, but ulcerated plaques and tumors alone do not establish separate visual value beyond existing cutaneous T cell lymphoma coverage.',[])
 if uid=='D1108':override('MERGE_EXISTING','Collapsing glomerulopathy in HIV is already within the focal segmental glomerulosclerosis spectrum; enlarged echogenic kidneys are nonspecific. Add the HIV associated clinical and imaging context to that page.',['focal_segmental_glomerulosclerosis'])
 if uid=='D1143':override('DEFER','Distinct acquired metabolic cause, but stones and nephrocalcinosis are nonspecific. Teach malabsorption associated hyperoxaluria in stone and nephrocalcinosis differentials.',[])
 if uid=='D1151':override('DEFER','Reninoma is distinct, but the proposed small cortical mass is nonspecific and the useful discriminator is biochemical renin excess.',[])
 if x.get('index')==35 and x.get('review_file')=='review_cardiology.json':
  override('DEFER','Left ventricular hypertrabeculation is a morphologic trait rather than a distinct cardiomyopathy in the 2023 ESC framework. Avoid presenting physiologic trabeculation as a disease; teach the pattern within its clinical cardiomyopathy context.',[])
  x['source_urls']=['https://academic.oup.com/eurheartj/article/44/37/3503/7246608']
 if x.get('index')==31 and x.get('review_file')=='review_cardiology.json':
  x['visual_anchor']='Restricted aortic leaflet opening with Doppler flow acceleration; include calcific adult stenosis and congenital valve morphology as clearly labeled forms.'
 if x.get('index')==28 and x.get('review_file')=='review_cardiology.json':
  x['visual_anchor']='Interatrial communication with Doppler shunting and right sided volume loading; show secundum and sinus venosus anatomy and label associated anomalous pulmonary veins.'
 if x['final_id']=='nutcracker_syndrome':
  x['visual_anchor']='Left renal vein compression with a beak and gonadal or pelvic collaterals on Doppler or CT, paired with compatible hematuria or pain; distinguish asymptomatic compression.'
 if x['final_id']=='superior_semicircular_canal_dehiscence_syndrome':
  x['visual_anchor']='Canal plane temporal bone CT shows a superior canal roof defect, paired with compatible sound or pressure induced symptoms and physiologic findings.'
 if norm(x['canonical_name'])==norm('PLCG2-associated disease (PLAID / APLAID)'):
  x['visual_anchor']='PLAID can show evaporative cold urticaria and neonatal acral ulcers; APLAID more often has blistering or inflammatory skin lesions without cold induced symptoms.'
  x['source_urls']=['https://www.nejm.org/doi/full/10.1056/NEJMoa1102140','https://pmc.ncbi.nlm.nih.gov/articles/PMC5664155/'];x['evidence_status']='supported'
 if norm(x['canonical_name'])==norm('Transitional ankle fractures'):
  x['visual_anchor']='CT distinguishes Tillaux fracture of the anterolateral distal tibial epiphysis from triplane fracture involving epiphysis, physis, and metaphysis in three planes.'
 if norm(x['canonical_name'])==norm('Pulmonary Talcosis'):
  x['visual_anchor']='CT after intravenous injection of talc containing crushed tablets may show centrilobular micronodules, high attenuation deposits, and later fibrosis or emphysema.'
 if norm(x['canonical_name'])==norm('Rhabdoid Tumor of the Kidney'):
  x['visual_anchor']='Large heterogeneous infant renal mass with necrosis or hemorrhage; subcapsular fluid or calcification may occur, with a recognized association with CNS rhabdoid tumors.'
 if norm(x['canonical_name'])==norm('Median Arcuate Ligament Syndrome'):
  x['visual_anchor']='Respiratory dependent hooked proximal celiac artery narrowing on sagittal CTA or Doppler in a compatible symptomatic syndrome; isolated compression can be incidental.'
 if uid=='D1019':override('MERGE_EXISTING','Broaden the existing Alpha Thalassemia page beyond HbH disease and include the Hb Barts hydrops phenotype; do not create parallel genotype specific pages.',['hemoglobin_h_disease'])
 if norm(x['canonical_name'])==norm('Mammary Myofibroblastoma'):
  override('DEFER','The proposed circumscribed breast mass overlaps common benign tumors; definitive separation is histologic and adds limited independent visual teaching value.',[])
 if norm(x['canonical_name']) in {norm(y) for y in ['Hard-Metal Lung Disease (Giant Cell Interstitial Pneumonia)','Pulmonary Inflammatory Myofibroblastic Tumor','Congenital Pulmonary Lymphangiectasia','Idiopathic Pulmonary Arterial Hypertension']}:
  override('DEFER','The supplied clinical or imaging appearance is nonspecific; the defining exposure, histology, or hemodynamic result does not provide a strong independent visual teaching pattern.',[])
 if norm(x['canonical_name'])==norm('Nodular Pulmonary Amyloidosis'):
  override('MERGE_EXISTING','Broaden the existing localized tracheobronchial amyloidosis topic to localized pulmonary amyloidosis and add the nodular form; preserve different radiologic distributions within that disease.',['tracheobronchial_amyloidosis'])
 if x['final_id']=='genetic_surfactant_dysfunction':
  override('ADD','One pediatric diffuse lung disease topic with genetic subtypes; do not publish separate near identical scan galleries for each surfactant gene.',[])
  x['visual_anchor']='Diffuse ground glass, septal thickening, and variable cystic change in term neonates or children; label SFTPB, SFTPC, ABCA3, and NKX2-1 subtypes and do not infer genotype from CT.'
 if x['final_id']=='asbestos_related_nonmalignant_thoracic_disease':
  x['visual_anchor']='Basal subpleural fibrosis for asbestosis; pleural plaques, diffuse pleural thickening, and associated rounded atelectasis as distinct nonmalignant manifestations.'
 if x['final_id']=='inert_metal_pneumoconiosis':
  x['visual_anchor']='Dense fine pulmonary nodules after iron, tin, or barium dust exposure; keep siderosis, stannosis, and baritosis as labeled etiologic sections.'
 if x['final_id']=='pulmonary_agenesis_and_aplasia':
  x['visual_anchor']='Absent or rudimentary ipsilateral main bronchus with absent lung tissue and vasculature, ipsilateral volume loss, and compensatory contralateral expansion.'
 if norm(x['canonical_name'])==norm('Sanjad-Sakati Syndrome'):
  x['visual_anchor']='Severe growth deficiency and characteristic face with deep set eyes, beaked nose, micrognathia, and congenital hypoparathyroidism; skeletal findings vary.'
  x['source_urls']=['https://pubmed.ncbi.nlm.nih.gov/30080992/']
 if uid=='D1046':
  x['visual_anchor']='Vesiculopustular lesions in a neonate with Down syndrome, sometimes concentrated at adhesive or venipuncture sites; blood evaluation establishes transient abnormal myelopoiesis.'
 if uid=='D1536':x['priority']='P1'
 if x['final_id']=='choroid_plexus_tumors':x['visual_anchor']='Frond like avidly enhancing ventricular papilloma versus invasive heterogeneous carcinoma with necrosis or dissemination; hydrocephalus may accompany either.'
 if x['final_id']=='hair_thread_tourniquet_syndrome':x['visual_anchor']='A constricting hair or thread with distal swelling or ischemia on a digit, toe, or penis; include genital presentations within one topic.'
 if x['final_id']=='physical_child_abuse':x['visual_anchor']='Classic metaphyseal lesions, suspicious rib fracture patterns, and intracranial injury; interpret the entire clinical context rather than treating an isolated sign as proof.'
 if x['final_id']=='splanchnic_venous_thrombosis':x['visual_anchor']='Portal, mesenteric, or splenic venous filling defects and chronic portal cavernous transformation; include infective pylephlebitis as a subtype.'
 if x['final_id']=='atelosteogenesis':x['visual_anchor']='Severe limb shortening with hypoplastic or poorly ossified long bones and characteristic joint or skeletal abnormalities; include boomerang phenotype as a labeled variant.'
 if x['final_id']=='perinatal_arterial_ischemic_stroke':
  override('MERGE_NEW','Neonatal arterial stroke is an age specific presentation of ischemic stroke. Use a perinatal section within the new acute ischemic stroke page.',[]);x['merge_target_id']='acute_ischemic_stroke'
 if x['final_id']=='choroid_plexus_tumors' and x['action'].startswith('MERGE'):
  x['action']='MERGE_NEW';x['merge_target_id']='choroid_plexus_tumors'
 if x.get('index')==57 and x.get('review_file')=='review_cardiology.json':
  x['action']='MERGE_NEW';x['merge_target_id']='aortic_stenosis'
 if x['final_id']=='malt_lymphoma':
  override('MERGE_EXISTING','The existing conjunctival lymphoma topic already explicitly lists MALT lymphoma. Broaden and reorganize it as extranodal MALT lymphoma with gastric and ocular sections, preserving any other ocular lymphoma types separately.',['conjunctival_lymphoma'])
 # Separate related proposed records from truly existing records.
 x['related_proposed_ids']=[q for q in x['existing_parent_ids'] if q not in EIDS]
 x['existing_parent_ids']=[q for q in x['existing_parent_ids'] if q in EIDS]
 if x['action']=='MERGE_EXISTING' and not x['existing_parent_ids'] and x['related_proposed_ids']:
  x['action']='MERGE_NEW';x['merge_target_id']=x['related_proposed_ids'][0]
# Split anatomically distinct rare hernias instead of one unrelated paired title.
split_rows=[]
for x in rows:
 if x.get('index')==118 and x.get('review_file')=='review_gastroenterology.json':
  for title,anchor in [('Spigelian hernia','Herniation through the Spigelian fascia along the semilunar line on CT or dynamic ultrasound.'),('Obturator hernia','Bowel extending through the obturator canal between the obturator and pectineus muscles on pelvic CT.')]:
   y=dict(x);y.update(final_name=title,final_id=slug(title),canonical_name=title,visual_anchor=anchor,decision_id=x['decision_id']+'_'+slug(title));split_rows.append(y)
 else:split_rows.append(x)
rows=split_rows
# Calibrate release order for the general physician and medical student audience.
promote={norm(x) for x in ['Horseshoe kidney','Acute pyelonephritis, including lobar nephronia','Renal and perinephric abscess','Renal osteodystrophy and secondary hyperparathyroidism','Erythema Toxicum Neonatorum','Neonatal mastitis','Umbilical Granuloma','Osgood-Schlatter disease','Lisfranc injury','Monteggia fracture-dislocation','Galeazzi fracture-dislocation','Calcaneus fracture','Patellar dislocation','Radial head fracture','Bladder calculus','Acute Myocarditis','Schatzki Ring','Omental Infarction','Pill-Induced Esophagitis','Acromioclavicular joint separation']}
demote={norm(x) for x in ['Pulmonary Artery Intimal Sarcoma','Alveolar Macrophage Pneumonia (formerly Desquamative Interstitial Pneumonia)','Diffuse Idiopathic Pulmonary Neuroendocrine Cell Hyperplasia (DIPNECH)','Combined Pulmonary Fibrosis and Emphysema','Bronchial Atresia','Neuroendocrine cell hyperplasia of infancy','Congenitally Corrected Transposition of the Great Arteries','Double Outlet Right Ventricle','Interrupted Aortic Arch','Partial Anomalous Pulmonary Venous Return','Heterotaxy syndrome','Pulmonary artery sling','Pulmonary Atresia with Ventricular Septal Defect','Hepatic sinusoidal obstruction syndrome','Atypical Teratoid/Rhabdoid Tumor','Currarino Syndrome','Congenital High Airway Obstruction Syndrome (CHAOS)']}
for x in rows:
 if norm(x['final_name']) in promote:x['priority']='P1'
 if norm(x['final_name']) in demote:x['priority']='P2'
# One distinct addition per final ID; consolidate independently reviewed mentions.
groups=collections.defaultdict(list)
for x in rows:groups[x['final_id']].append(x)
additions=[]
for key,mentions in groups.items():
 eligible=[x for x in mentions if x['action']=='ADD']
 if not eligible:continue
 primary=sorted(eligible,key=lambda x:(x['priority']!='P1',x['review_file']=='review_orthopedics.json' and x.get('uid') in ['D1674','D1676','D1678','D1680','D1682']))[0]
 out=dict(topic_id=key,name=primary['final_name'],specialty=primary['home_specialty'],priority='P1' if any(x['priority']=='P1' for x in eligible) else 'P2',visual_anchor=primary['visual_anchor'],inclusion_rationale=primary['rationale'],source_mentions=[dict(decision_id=x['decision_id'],uid=x.get('uid'),section=x['section'],name=x['canonical_name']) for x in mentions],cross_specialties=sorted({x['section'] for x in mentions}|{primary['home_specialty']}),source_urls=sorted({u for x in eligible for u in x['source_urls']}))
 additions.append(out)
additions.sort(key=lambda x:(x['specialty'],x['priority'],x['name'].lower()))
(BASE/'integrated_decisions.json').write_text(json.dumps(rows,indent=2,ensure_ascii=False))
(BASE/'integrated_additions.json').write_text(json.dumps(additions,indent=2,ensure_ascii=False))
coverage={x.get('uid') for x in rows};missing=[x for x in HEADINGS if x['uid'] not in coverage]
invalid=[(x['decision_id'],q) for x in rows for q in x['existing_parent_ids'] if q not in EIDS]
print('Decision records',len(rows),'unique candidate keys',len(groups),'ADD',len(additions),'priority',collections.Counter(x['priority'] for x in additions));print('Homes',collections.Counter(x['specialty'] for x in additions));print('Missing headings',[(x['uid'],x['name']) for x in missing]);print('Nonexisting cited ids',invalid)
print('Potential duplicate names:')
for i,a in enumerate(additions):
 for b in additions[i+1:]:
  score=difflib.SequenceMatcher(None,norm(a['name']),norm(b['name'])).ratio()
  if score>.86:print(round(score,2),a['name'],'<>',b['name'])
