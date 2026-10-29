import json,re,unicodedata
from pathlib import Path
p=Path('outputs/topic_review');h={x['uid']:x for x in json.loads((p/'all_headings.json').read_text())}
def slug(s):return re.sub(r'[^a-z0-9]+','_',unicodedata.normalize('NFKD',s).encode('ascii','ignore').decode().lower()).strip('_')
rows='''D1159|ADD|P1|Acute ischemic stroke|Territorial diffusion restriction, early CT ischemic changes, and arterial occlusion on CTA.|Core emergency with management relevant imaging.|
D1161|ADD|P1|Aneurysmal subarachnoid hemorrhage|Blood in basal cisterns and sulci on CT, with a causative aneurysm on CTA or angiography.|Distinct hemorrhagic emergency from intraparenchymal hemorrhage.|
D1163|ADD|P1|Spinal epidural abscess|Rim enhancing epidural collection with cord or cauda equina compression on MRI.|Distinct imageable spinal infection requiring urgent recognition.|
D1165|ADD|P1|Spinal cord infarction|Anterior cord diffusion restriction; sagittal pencil like signal and axial owl eyes may occur.|Named vascular cord disease; signs are supportive rather than uniquely diagnostic.|
D1167|MERGE_EXISTING|P1|Cauda equina syndrome|MRI demonstrates the compressive cause; clinical saddle sensory and sphincter findings define the emergency.|Add emergency presentation and red flags to lumbar disc herniation and stenosis pages, with cross links to tumors and infection.|lumbar_disc_herniation;lumbar_spinal_stenosis
D1169|ADD|P1|Guillain Barre syndrome|Facial diplegia or flaccid weakness; smooth cauda equina root enhancement can support an atypical case.|Core clinical syndrome; enhancement is nonspecific and MRI is not a routine confirmatory test.|
D1172|ADD|P1|Osmotic demyelination syndrome|Symmetric central pontine signal with peripheral sparing, sometimes with extrapontine lesions.|Recognizable toxic metabolic MRI pattern.|
D1174|ADD|P1|Carbon monoxide poisoning|Bilateral globus pallidus injury and delayed white matter abnormalities may appear on MRI.|Important toxic exposure with a memorable imaging differential; normal early imaging does not exclude it.|
D1176|ADD|P2|Infratentorial superficial siderosis|Susceptibility blooming coating cerebellar folia, brainstem, and cranial nerves.|Distinct distribution and chronic clinical syndrome; separate from cortical siderosis in cerebral amyloid angiopathy.|
D1178|MERGE_EXISTING|P2|Balo concentric sclerosis|Alternating concentric rings on T2 and diffusion imaging.|Teach the concentric demyelination phenotype within multiple sclerosis rather than a parallel disease page.|multiple_sclerosis
D1180|ADD|P2|CLIPPERS|Numerous punctate and curvilinear enhancing foci centered on the pons and cerebellar peduncles.|Recognizable inflammatory syndrome; label as a diagnosis requiring exclusion of mimics, especially lymphoma.|
D1182|ADD|P2|Marchiafava Bignami disease|Corpus callosum swelling or injury with selective central layer involvement.|Distinct alcohol or malnutrition associated callosal disease.|
D1184|ADD|P2|Metronidazole induced encephalopathy|Symmetric dentate nucleus abnormalities, sometimes involving the splenium or brainstem.|Characteristic medication associated imaging pattern.|
D1186|ADD|P2|Neuronal intranuclear inclusion disease|Curvilinear DWI hyperintensity along the corticomedullary junction.|Strong imaging clue beyond the confirmatory skin biopsy; not present in every case.|
D1188|ADD|P2|Biotin thiamine responsive basal ganglia disease|Symmetric caudate and putamen swelling or injury on MRI in acute encephalopathy.|Treatable metabolic disorder with an important imaging trigger; genetic confirmation is needed.|
D1190|ADD|P2|HTRA1 related cerebral small vessel disease|Leukoencephalopathy with lacunes; classic CARASIL adds early alopecia and spondylosis, and an arc sign can involve the pons and middle cerebellar peduncles.|Distinct from CADASIL; use HTRA1 spectrum title and CARASIL as a searchable phenotype.|
D1192|ADD|P2|Japanese encephalitis|Bilateral thalamic lesions, sometimes involving basal ganglia and brainstem.|Imageable neuroinfection; pattern requires epidemiologic and laboratory context.|
D1194|ADD|P2|Listeria rhombencephalitis|Brainstem and cerebellar peduncle inflammation or microabscesses on MRI.|Distinctive hindbrain distribution makes a useful neuroinfection topic.|
D1196|ADD|P1|Acute flaccid myelitis|Longitudinal gray matter predominant spinal cord T2 signal with acute asymmetric flaccid weakness.|Important pediatric syndrome distinct from poliomyelitis.|
D1198|MERGE_EXISTING|P2|Subacute sclerosing panencephalitis|Progressive white matter abnormalities and atrophy are nonspecific; EEG and virologic evidence establish the syndrome.|Add late neurologic complication coverage within measles; the proposal does not establish a distinctive independent visual signature.|measles
D1201|ADD|P1|Kernicterus|Bilateral pallidal signal abnormalities, with chronic T2 changes; early bilirubin injury can be T1 bright.|Core neonatal toxic brain injury; state sequence and disease stage correctly.|
D1203|ADD|P1|Germinal matrix and intraventricular hemorrhage of prematurity|Caudothalamic groove hemorrhage and intraventricular blood on cranial ultrasound.|Distinct neonatal hemorrhage pattern; include periventricular hemorrhagic infarction as an associated venous injury.|
D1205|ADD|P1|Periventricular leukomalacia|Periventricular cysts or white matter volume loss with irregular ventricular margins.|Classic injury of prematurity, distinct from term neonatal hypoxic ischemic encephalopathy.|
D1207|ADD|P1|Polymicrogyria|Excess small gyri with an irregular cortical surface and gray white junction on MRI.|Imaging defined cortical malformation; keep regional forms together.|
D1209|ADD|P1|Periventricular nodular heterotopia|Gray matter signal nodules lining the lateral ventricles.|Distinct migration disorder from subcortical band heterotopia.|
D1211|ADD|P1|Ependymoma|Intraventricular or spinal mass; posterior fossa tumors may extend through the fourth ventricular outlets.|Major missing neurooncology topic; organize by current molecular and anatomic classification, not an assertion that every type is the same tumor.|
D1213|ADD|P1|Oligodendroglioma|Cortical or subcortical infiltrative mass often containing calcification.|Important brain tumor differential; imaging cannot establish IDH mutation or 1p19q codeletion.|
D1215|ADD|P2|Ganglioglioma|Superficial cortical tumor, often temporal, sometimes cystic with a mural nodule or calcification.|Distinct epilepsy associated glioneuronal tumor; separate from DNET.|
D1215|ADD|P2|Dysembryoplastic neuroepithelial tumor|Bubbly multinodular cortical lesion with little mass effect in a patient with epilepsy.|Separate tumor with a useful MRI pattern; do not combine with ganglioglioma.|
D1217|ADD|P2|Neuralgic amyotrophy|Scapular winging with patchy muscle denervation edema or atrophy on MRI.|Recognizable clinical and imaging syndrome, distinct from entrapment neuropathy.|
D1219|ADD|P1|Frontotemporal dementia|Frontal and anterior temporal predominant atrophy or hypometabolism.|Major dementia syndrome with imageable regional patterns; keep phenotypes within one topic.|
D1224|ADD|P2|Heroin inhalation associated leukoencephalopathy|Symmetric cerebellar and posterior cerebral white matter abnormalities.|Characteristic exposure related distribution; preserve common chasing the dragon search term.|
D1226|MERGE_EXISTING|P2|Copper deficiency myeloneuropathy|Symmetric dorsal column cord signal may mimic vitamin B12 related subacute combined degeneration.|Teach as an etiologic differential in the existing dorsal column syndrome page; no independent visual discriminator is supplied.|subacute_combined_degeneration
D1226|MERGE_EXISTING|P1|Nitrous oxide induced myeloneuropathy|Dorsal column inverted V signal from functional vitamin B12 inactivation.|Add exposure specific cause and clinical teaching to existing subacute combined degeneration.|subacute_combined_degeneration
D1228|ADD|P2|CSF1R related leukoencephalopathy|Confluent white matter injury, callosal thinning, persistent diffusion abnormalities, and small CT calcifications.|Distinct adult white matter disorder; use current spectrum name and retain ALSP as a search alias.|
D1230|ADD|P2|Nasu Hakola disease|Cystic bone lesions paired with frontal predominant white matter degeneration.|Distinct combined skeletal and neurologic syndrome.|
D1232|ADD|P2|Retinal vasculopathy with cerebral leukoencephalopathy and systemic manifestations|Retinal vascular abnormalities and enhancing white matter lesions that can resemble tumors.|Multisystem TREX1 associated syndrome; separate from Aicardi Goutieres syndrome.|
D1234|ADD|P2|Beta propeller protein associated neurodegeneration|Iron sensitive substantia nigra and pallidal signal, with a T1 bright nigral halo in later disease.|Distinct NBIA phenotype with strong MRI teaching value.|
D1236|ADD|P2|Alpers Huttenlocher syndrome|Posterior cortical and thalamic MRI abnormalities with refractory epilepsy and liver disease.|Recognizable POLG related clinicoradiologic phenotype; findings are not specific alone.|
D1238|ADD|P2|Acute necrotizing encephalopathy|Symmetric thalamic lesions, sometimes with concentric hemorrhagic or necrotic components.|Distinct infection triggered encephalopathy rather than infectious encephalitis itself.|
D1240|ADD|P2|West Nile neuroinvasive disease|Deep gray, brainstem, or anterior horn abnormalities can accompany encephalitis or acute flaccid paralysis.|Useful neuroinfectious differential; MRI may be normal and laboratory confirmation is essential.|
D1242|MERGE_EXISTING|P2|Enterovirus A71 brainstem encephalitis|Dorsal brainstem signal with a compatible hand foot and mouth eruption.|Add neurologic disease pattern to the existing hand foot and mouth topic and cross link acute flaccid myelitis.|hand_foot_and_mouth_disease
D1244|DEFER|P2|Primary amebic meningoencephalitis|Diffuse edema and meningeal enhancement lack a reliable distinguishing imaging pattern.|The proposal relies on organism microscopy and late nonspecific necrosis, insufficient for the current clinical and radiologic scope.|
D1246|ADD|P2|Idiopathic hypertrophic pachymeningitis|Focal or diffuse thickened enhancing dura on MRI.|Imageable inflammatory dural syndrome; idiopathic label requires exclusion of infection, malignancy, and systemic disease.|
D1248|ADD|P2|Autoimmune GFAP astrocytopathy|Radial perivascular enhancement extending from the ventricles, with possible cord or meningeal involvement.|Distinct autoimmune syndrome from genetic GFAP associated Alexander disease; imaging is supportive.|
D1250|MERGE_EXISTING|P2|Intravascular large B cell lymphoma|Multifocal infarct like lesions of different ages or other variable brain MRI abnormalities.|Existing Hematology topic already names the same disease; expand neurologic coverage.|intravascular_large_b_cell_lymphoma
D1252|ADD|P2|Dyke Davidoff Masson syndrome|Cerebral hemiatrophy with ipsilateral calvarial thickening and sinus enlargement.|Coherent clinicoradiologic syndrome with a memorable structural pattern.|
D1254|ADD|P2|COL4A1 related disorders|Porencephaly, small vessel disease, and variable ocular abnormalities such as cataract.|Distinct genetic multisystem disease spectrum; avoid separate pages for every manifestation.|
D1256|ADD|P1|Hydranencephaly|Near absence of cerebral hemispheres replaced by fluid, with preserved falx and variable deep structures.|Distinct severe malformation from hydrocephalus and holoprosencephaly.|
D1258|ADD|P2|Horizontal gaze palsy with progressive scoliosis|Split pons and butterfly shaped medulla with progressive scoliosis.|Distinct structural brainstem and skeletal syndrome.|
D1260|ADD|P2|Rhombencephalosynapsis|Fusion of cerebellar hemispheres across a deficient vermis.|Imaging defined hindbrain malformation; syndromic associations belong within the topic.|
D1262|ADD|P2|Pontocerebellar hypoplasia|Small ventral pons and cerebellum; a dragonfly configuration occurs in some forms.|Use a single clinicoradiologic family topic with genetic subtypes; no claim that the dragonfly pattern occurs in all forms.|
D1264|ADD|P2|LAMA2 related muscular dystrophy|Congenital weakness or contractures with extensive characteristic white matter signal abnormalities.|Distinct muscle and brain phenotype, not covered by Walker Warburg or Ullrich disease.|
D1266|ADD|P2|Idiopathic spinal cord herniation|Focal ventral cord displacement and kinking through a dural defect on MRI.|Distinct structural cause of progressive myelopathy.|
D1268|ADD|P2|Spinal arachnoid web|Focal dorsal cord indentation producing a sagittal scalpel sign, sometimes with a syrinx.|Specific structural lesion; distinguish from ventral cord herniation and arachnoid cyst.|
D1270|ADD|P1|Intracranial epidermoid cyst|Extra axial CSF like lesion with marked DWI hyperintensity.|Distinct intracranial developmental lesion from cutaneous epidermal inclusion cyst.|
D1272|ADD|P2|Intracranial dermoid cyst|Fat containing intracranial lesion, with disseminated fat droplets after rupture.|Distinct developmental lesion with a strong CT and MRI anchor.|
D1274|ADD|P2|Central neurocytoma|Bubbly intraventricular mass near the foramen of Monro and septum pellucidum.|Useful age and location based tumor pattern.|
D1276|ADD|P2|Choroid plexus papilloma|Lobulated avidly enhancing intraventricular frond like mass with hydrocephalus.|Distinct ventricular tumor from ependymoma; imaging alone does not establish benignity.|
D1278|ADD|P2|Pleomorphic xanthoastrocytoma|Superficial hemispheric cyst and mural nodule, sometimes with meningeal contact.|Named cortical tumor with useful age, epilepsy, and location context.|
D1281|ADD|P2|SPG11 related hereditary spastic paraplegia|Thin corpus callosum and ears of the lynx white matter signal near the frontal horns.|Distinct MRI supported spastic paraplegia; sign also occurs in SPG15 and is not uniquely diagnostic.|
D1283|MERGE_EXISTING|P2|Becker muscular dystrophy|Calf pseudohypertrophy and muscle fatty replacement overlap Duchenne disease.|Broaden existing Duchenne page to dystrophinopathies and show Becker comparison; a separate near identical visual page adds little.|duchenne_muscular_dystrophy
D1285|ADD|P2|Dysferlinopathy|Miyoshi phenotype shows prominent posterior calf wasting, with selective muscle involvement on imaging.|Use a single DYSF spectrum topic including Miyoshi and limb girdle presentations.|
D1287|ADD|P2|X linked myotubular myopathy|Long myopathic face, ptosis, ophthalmoparesis, and severe neonatal hypotonia.|Visible phenotype supports a clinical atlas entry beyond its biopsy finding.|
D1289|DEFER|P2|Central core disease|Hypotonia is nonspecific; central cores are a muscle histology finding.|No sufficiently distinctive clinical or radiologic anchor is supplied.|
D1291|DEFER|P2|Hereditary neuropathy with liability to pressure palsies|Recurrent focal palsies have no stable distinctive photograph; tomacula require nerve histology.|Primarily electrodiagnostic and genetic diagnosis with weak standalone visual value.|
D1293|DEFER|P2|Multifocal motor neuropathy|Asymmetric hand wasting and nerve enlargement are nonspecific.|Prioritize stronger phenotype and image based additions; diagnosis depends on electrophysiology and clinical context.|
D1295|DEFER|P2|Lafora disease|PAS positive sweat gland inclusions are histologic.|The proposed teaching image is pathology rather than a clinical or radiologic manifestation.|
D1297|DEFER|P2|MERRF syndrome|Ragged red fibers are histologic and multiple lipomas are variable and nonspecific.|Insufficient independent clinical or radiologic anchor for current release.|
D1299|DEFER|P2|PLA2G6 associated infantile neuroaxonal dystrophy|Cerebellar atrophy alone is nonspecific; pallidal iron may appear later.|Defer the proposed infantile phenotype until a distinct reproducible clinical and MRI image set is demonstrated.|
D1301|ADD|P2|Familial dysautonomia|Smooth tongue from absent fungiform papillae and absent overflow emotional tears.|Directly photographable diagnostic phenotype distinct from congenital insensitivity to pain with anhidrosis.|
D1425|MERGE_EXISTING|P1|Erosive hand osteoarthritis|Central interphalangeal erosions with gull wing or saw tooth configuration.|Add an erosive phenotype section to existing osteoarthritis; new imaging content does not require a new disease page.|osteoarthritis
D1427|ADD|P1|Secondary hypertrophic osteoarthropathy|Digital clubbing with symmetric long bone periostosis and parallel cortical scintigraphic uptake.|Useful acquired syndrome across multiple causes; distinct from primary pachydermoperiostosis.|
D1429|ADD|P1|Neuropathic arthropathy|Joint fragmentation, debris, disorganization, and collapse; rocker bottom foot in advanced midfoot disease.|Coherent image defined syndrome across etiologies; keep all joints and causes together.|
D1432|ADD|P2|Transient osteoporosis of the hip|Diffuse femoral head and neck marrow edema, later osteopenia, without the typical focal necrotic boundary.|Important clinical and imaging differential from osteonecrosis and insufficiency fracture.|
D1434|ADD|P2|Dialysis related beta 2 microglobulin amyloidosis|Periarticular bone cysts and destructive spondyloarthropathy after long term dialysis.|Distinct precursor protein and distribution from systemic AL amyloidosis.|
D1436|ADD|P2|Palmar fasciitis and polyarthritis syndrome|Rapidly progressive woody palmar thickening with flexion contractures and polyarthritis.|Distinct photographable paraneoplastic syndrome; retain underlying malignancy associations in the page.|
D1438|ADD|P2|Pachydermodactyly|Painless lateral proximal interphalangeal soft tissue thickening without erosions.|Useful visible mimic of inflammatory arthritis.|
D1440|ADD|P2|Camptodactyly arthropathy coxa vara pericarditis syndrome|Camptodactyly, noninflammatory joint swelling, and coxa vara with short femoral necks.|Distinct PRG4 associated clinical and skeletal phenotype.|
D1442|ADD|P2|Multicentric osteolysis nodulosis and arthropathy|Carpal and tarsal osteolysis with palmoplantar nodules and contractures.|Canonical MMP2 related MONA topic; Torg and NAO belong here, while MMP14 Winchester syndrome is separate.|
D1442|DEFER|P2|MMP14 related Winchester syndrome|Progressive osteolysis and contractures overlap MONA.|Keep separate in disease ontology; defer a second closely overlapping visual page until its own useful clinical and imaging set is demonstrated.|
D1444|ADD|P2|COPA syndrome|Childhood diffuse lung disease with ground glass, cysts, and nodules paired with arthritis.|Distinct inherited immune dysregulation syndrome; place in Immunology with rheumatology and pulmonology cross links.|
D1446|ADD|P2|Singleton Merten syndrome|Early aortic or valve calcification, abnormal dentition, and acro osteolysis.|Strong multisystem radiographic phenotype.|
D1448|ADD|P2|Stiff skin syndrome|Indurated bound down skin with hypertrichosis and joint contractures.|Directly visible fibrosing disorder distinct from stiff person syndrome and systemic sclerosis.|
D1450|ADD|P2|Fibroblastic rheumatism|Periarticular cutaneous nodules with rapidly developing flexion contractures.|Recognizable clinicocutaneous syndrome; biopsy supports diagnosis.|
D1452|ADD|P2|Voriconazole induced periostitis|Irregular bulky periosteal new bone, often involving ribs, clavicles, or scapulae.|Distinct medication associated radiographic phenotype from smoother secondary hypertrophic osteoarthropathy.|
'''
refs={
'D1169':['https://onlinelibrary.wiley.com/doi/full/10.1111/ene.16073'],
'D1186':['https://pmc.ncbi.nlm.nih.gov/articles/5382941/'],
'D1188':['https://www.ncbi.nlm.nih.gov/books/NBK169615/'],
'D1190':['https://www.ncbi.nlm.nih.gov/sites/books/NBK32533/'],
'D1228':['https://www.ncbi.nlm.nih.gov/books/NBK100239/'],
'D1234':['https://www.ncbi.nlm.nih.gov/books/NBK424403/'],
'D1248':['https://www.ajnr.org/content/ajnr/early/2024/06/06/ajnr.A8236.full.pdf'],
'D1264':['https://www.ncbi.nlm.nih.gov/books/NBK97333/'],
'D1268':['https://pmc.ncbi.nlm.nih.gov/articles/PMC7964642/'],
'D1281':['https://pubmed.ncbi.nlm.nih.gov/30606727/'],
'D1285':['https://www.ncbi.nlm.nih.gov/books/NBK1303/'],
'D1287':['https://www.ncbi.nlm.nih.gov/books/NBK1432/'],
'D1301':['https://www.ncbi.nlm.nih.gov/books/NBK1180/'],
'D1442':['https://www.ncbi.nlm.nih.gov/books/NBK373578/'],
'D1444':['https://pubmed.ncbi.nlm.nih.gov/36746811/'],
'D1452':['https://pmc.ncbi.nlm.nih.gov/articles/PMC3888175/']}
out=[]
for line in rows.strip().splitlines():
 uid,action,priority,name,anchor,reason,parent=line.split('|')
 out.append(dict(uid=uid,section=h[uid]['section'],proposed_id=slug(name),canonical_name=name,action=action,priority=priority,visual_anchor=anchor,rationale=reason,existing_parent_ids=parent.split(';') if parent else [],source_urls=refs.get(uid,[])))
(p/'review_neuro_rheum.json').write_text(json.dumps(out,indent=2))
print('Written',len(out))
