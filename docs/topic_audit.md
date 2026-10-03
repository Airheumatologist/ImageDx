# Topic audit and image-library expansion

Audited 2026-10-03 against the image-library rule:
a topic is a disease or syndrome that produces clinical images.
General non-visual diseases stay out. A single manifestation is not a topic.

- Active topics after both passes: 495
- Excluded from the active index: 73 (retained in `excluded_topics`)
- Added in the first pass: 63; second pass: 52

## Counts by specialty

| Specialty | Active | Added | Excluded |
| --- | ---: | ---: | ---: |
| Cardiology | 25 | 4 | 13 |
| Dermatology | 58 | 12 | 1 |
| Endocrinology | 20 | 3 | 7 |
| Gastroenterology | 29 | 3 | 12 |
| Hematology | 29 | 2 | 1 |
| Infectious Diseases | 52 | 8 | 2 |
| Nephrology | 16 | 3 | 11 |
| Neurology | 40 | 3 | 7 |
| Ophthalmology | 26 | 8 | 6 |
| Orthopedics | 46 | 6 | 3 |
| Pediatrics | 38 | 4 | 0 |
| Pulmonology | 21 | 3 | 10 |
| Rheumatology | 43 | 4 | 0 |

## Excluded

- `cardiac_papillary_fibroelastoma` (Cardiology): Echo and histology only; no clinical photograph.
- `chagas_cardiomyopathy` (Cardiology): Cardiac manifestation of Chagas disease, which is already an infectious-disease topic.
- `cor_triatriatum_sinistrum` (Cardiology): Imaging diagnosis (echo/CT) without a disease-specific external image.
- `pericardial_teratoma` (Cardiology): Fetal/surgical imaging mass without a bedside clinical image.
- `left_ventricular_noncompaction` (Cardiology): Echocardiographic/MRI diagnosis without pathognomonic clinical photographs.
- `libman_sacks_endocarditis` (Cardiology): Manifestation of SLE / antiphospholipid syndrome, both already topics. Valvular vegetations are a finding, not a separate image-library disease.
- `nonbacterial_thrombotic_endocarditis` (Cardiology): Manifestation of malignancy or antiphospholipid syndrome, not an independent photographic disease.
- `patent_ductus_arteriosus_eisenmenger` (Cardiology): Eisenmenger physiology is a complication of shunt lesions, not a standalone topic.
- `persistent_left_superior_vena_cava` (Cardiology): Anatomic venous variant. Diagnosis is catheter or CT position, with no clinical photograph.
- `pulmonary_artery_sarcoma` (Cardiology): CT/PET vascular filling defect without a clinical photograph.
- `sinus_of_valsalva_aneurysm` (Cardiology): Echo/CT/angiographic lesion without a disease-specific photograph.
- `takotsubo_cardiomyopathy` (Cardiology): Ventriculography diagnosis. Presentation is chest pain, not a visual finding.
- `uhls_anomaly` (Cardiology): Rare imaging/pathology diagnosis of a parchment right ventricle, without clinical photographs.
- `necrolytic_migratory_erythema` (Dermatology): Cutaneous manifestation of glucagonoma, already an endocrinology topic.
- `insulinoma` (Endocrinology): Hypoglycemia syndrome. Whipple triad is not a visual finding.
- `nelson_syndrome` (Endocrinology): Post-adrenalectomy pituitary manifestation of treated Cushing disease, not a primary image topic.
- `pituitary_apoplexy` (Endocrinology): Acute hemorrhagic manifestation of a pituitary adenoma, not a separate disease.
- `primary_aldosteronism` (Endocrinology): Endocrine hypertension without clinical photographs. Same exclusion class as essential hypertension.
- `sheehan_syndrome` (Endocrinology): Postpartum pituitary failure without a photographic phenotype.
- `thyroid_eye_disease` (Endocrinology): Orbitopathy is a manifestation of Graves disease, already indexed with dermopathy and eye disease.
- `vipoma_syndrome` (Endocrinology): Secretory-diarrhea syndrome without a disease-specific photograph.
- `adult_intussusception` (Gastroenterology): Mechanical event, usually secondary to a lead-point tumor already covered elsewhere.
- `autoimmune_pancreatitis` (Gastroenterology): Pancreatic manifestation of IgG4-related disease, already a rheumatology topic.
- `cecal_volvulus` (Gastroenterology): Radiographic emergency without a disease-specific clinical photograph.
- `focal_nodular_hyperplasia` (Gastroenterology): Incidental liver imaging lesion. No clinical photograph.
- `gallstone_ileus_bouveret` (Gastroenterology): Fistula complication of gallstone disease, not a primary disease.
- `gastric_volvulus` (Gastroenterology): Radiographic/endoscopic emergency without a disease phenotype.
- `hepatic_hemangioma_giant` (Gastroenterology): Imaging lesion without a clinical photograph (distinct from cutaneous infantile hemangioma).
- `hepatic_adenoma` (Gastroenterology): Imaging lesion. Rupture is a complication, not a visual disease phenotype.
- `intraductal_papillary_mucinous_neoplasm` (Gastroenterology): Pancreatic cyst diagnosis by MRI/EUS, without a clinical photograph.
- `mirizzi_syndrome` (Gastroenterology): Biliary compression complication of gallstone disease.
- `sigmoid_volvulus` (Gastroenterology): Radiographic emergency without a disease-specific clinical photograph.
- `superior_mesenteric_artery_syndrome` (Gastroenterology): Compression anatomy on barium/CT, not a photographic disease.
- `hemophilic_arthropathy` (Hematology): Joint manifestation of hemophilia. Hemophilia A is added as the disease topic.
- `osteomyelitis` (Infectious Diseases): Umbrella bone infection. Image search returns mixed organisms, implants, and diagrams rather than one disease phenotype.
- `septic_arthritis` (Infectious Diseases): Umbrella joint infection, not a single visual disease.
- `bartter_syndrome` (Nephrology): Electrolyte tubulopathy. Failure to thrive is not disease-specific imagery.
- `complete_ureteral_duplication_ureterocele` (Nephrology): Congenital imaging anatomy; the clinical clue (prolapsed ureterocele) is a finding, not the topic grain wanted here.
- `congenital_nephrogenic_diabetes_insipidus` (Nephrology): Water-balance disorder without disease-specific images. Same exclusion class as diabetes insipidus generally.
- `dent_disease` (Nephrology): Low-molecular-weight proteinuria tubulopathy without clinical photographs.
- `emphysematous_pyelonephritis` (Nephrology): CT gas diagnosis of a severe infection, not a photographic disease.
- `encapsulating_peritoneal_sclerosis` (Nephrology): Dialysis complication rather than a primary visual disease.
- `fibromuscular_dysplasia` (Nephrology): Angiographic string-of-beads finding without a disease-specific photograph.
- `medullary_sponge_kidney` (Nephrology): Urographic diagnosis (paintbrush papillae) without clinical photographs.
- `nutcracker_syndrome` (Nephrology): Venous compression anatomy on Doppler/CT, not a photographic disease.
- `renal_papillary_necrosis` (Nephrology): Papillary complication of sickle cell disease, analgesic use, or diabetes.
- `von_hippel_lindau_disease` (Nephrology): Duplicate of the neurology von Hippel-Lindau topic.
- `balo_concentric_sclerosis` (Neurology): MRI pattern / variant of demyelination, not an independent clinical-image disease.
- `clippers_syndrome` (Neurology): Punctate MRI pattern with brainstem signs, without a photographic phenotype.
- `central_pontine_myelinolysis` (Neurology): MRI diagnosis of an osmotic demyelination syndrome. No clinical photograph.
- `cadasil` (Neurology): MRI leukoencephalopathy and lacunar disease. No disease-specific clinical photograph.
- `neurosarcoidosis` (Neurology): Neurologic manifestation of sarcoidosis, already a pulmonology topic.
- `progressive_multifocal_leukoencephalopathy` (Neurology): MRI and CSF diagnosis. Clinical deficits are not photographable disease findings.
- `superficial_siderosis` (Neurology): MRI hemosiderin staining. Hearing loss and ataxia are not photographic.
- `branch_retinal_vein_occlusion` (Ophthalmology): Retinal vascular event, usually a manifestation of systemic vascular disease rather than a disease entity.
- `brown_syndrome` (Ophthalmology): Restricted-motility finding, not a disease with a gallery of clinical images.
- `central_retinal_artery_occlusion` (Ophthalmology): Ophthalmic vascular event / manifestation, not a disease-level topic.
- `central_retinal_vein_occlusion` (Ophthalmology): Retinal vascular event / manifestation, not a disease-level topic.
- `horner_syndrome` (Ophthalmology): Pupillary/oculosympathetic manifestation of several diseases (dissection, Pancoast, cluster headache), not a disease.
- `proliferative_diabetic_retinopathy` (Ophthalmology): Retinal manifestation of diabetes mellitus. Diabetes was excluded as a general non-visual topic; its single fundus complication should not re-enter as its own topic.
- `freiberg_infraction` (Orthopedics): Osteochondrosis seen on foot radiographs, without a disease-specific photograph.
- `charcot_arthropathy` (Orthopedics): Neuroarthropathy is a manifestation of neuropathy (often diabetes or syphilis), not a primary disease.
- `osgood_schlatter_disease` (Orthopedics): Traction apophysitis. Swelling and a fragmented tubercle are a finding, not a disease gallery.
- `asbestosis` (Pulmonology): Occupational imaging/pleural-plaque diagnosis without a clinical photograph.
- `bronchopulmonary_sequestration` (Pulmonology): CT/angiographic congenital lesion without a clinical photograph.
- `congenital_lobar_emphysema` (Pulmonology): Neonatal radiographic diagnosis without a disease-specific photograph.
- `congenital_pulmonary_airway_malformation` (Pulmonology): Prenatal/CT cystic lung lesion without a clinical photograph.
- `mounier_kuhn_syndrome` (Pulmonology): Airway-caliber imaging diagnosis without a clinical photograph.
- `pancoast_tumor` (Pulmonology): Regional manifestation of apical lung cancer (Horner syndrome, hand wasting), not a separate disease.
- `scimitar_syndrome` (Pulmonology): Chest-radiograph vascular anomaly without a photographic phenotype.
- `silicosis` (Pulmonology): Occupational radiographic diagnosis (nodules, eggshell nodes) without a clinical photograph.
- `superior_vena_cava_syndrome` (Pulmonology): Obstructive manifestation of thoracic malignancy or thrombosis.
- `swyer_james_macleod_syndrome` (Pulmonology): Unilateral hyperlucent-lung radiographic diagnosis.

## Added

- `williams_syndrome` (Cardiology): Williams Syndrome
- `holt_oram_syndrome` (Cardiology): Holt-Oram Syndrome
- `loeys_dietz_syndrome` (Cardiology): Loeys-Dietz Syndrome
- `carcinoid_heart_disease` (Cardiology): Carcinoid Heart Disease
- `mycosis_fungoides` (Dermatology): Mycosis Fungoides
- `basal_cell_carcinoma` (Dermatology): Basal Cell Carcinoma
- `cutaneous_squamous_cell_carcinoma` (Dermatology): Cutaneous Squamous Cell Carcinoma
- `darier_disease` (Dermatology): Darier Disease
- `hailey_hailey_disease` (Dermatology): Hailey-Hailey Disease
- `lamellar_ichthyosis` (Dermatology): Lamellar Ichthyosis
- `xeroderma_pigmentosum` (Dermatology): Xeroderma Pigmentosum
- `cutaneous_mastocytosis` (Dermatology): Cutaneous Mastocytosis
- `lichen_planus` (Dermatology): Lichen Planus
- `vitiligo` (Dermatology): Vitiligo
- `lichen_sclerosus` (Dermatology): Lichen Sclerosus
- `infantile_hemangioma` (Dermatology): Infantile Hemangioma
- `primary_adrenal_insufficiency` (Endocrinology): Primary Adrenal Insufficiency
- `turner_syndrome` (Endocrinology): Turner Syndrome
- `multiple_endocrine_neoplasia_type_2a` (Endocrinology): Multiple Endocrine Neoplasia Type 2A
- `eosinophilic_esophagitis` (Gastroenterology): Eosinophilic Esophagitis
- `intestinal_amebiasis` (Gastroenterology): Intestinal Amebiasis
- `hepatocellular_carcinoma` (Gastroenterology): Hepatocellular Carcinoma
- `hemophilia_a` (Hematology): Hemophilia A
- `immune_thrombocytopenia` (Hematology): Immune Thrombocytopenia
- `cutaneous_leishmaniasis` (Infectious Diseases): Cutaneous Leishmaniasis
- `herpes_zoster` (Infectious Diseases): Herpes Zoster
- `mpox` (Infectious Diseases): Mpox
- `measles` (Infectious Diseases): Measles
- `diphtheria` (Infectious Diseases): Respiratory Diphtheria
- `yaws` (Infectious Diseases): Yaws
- `orf` (Infectious Diseases): Orf
- `hand_foot_and_mouth_disease` (Infectious Diseases): Hand, Foot, and Mouth Disease
- `goodpasture_syndrome` (Nephrology): Anti-GBM Disease
- `prune_belly_syndrome` (Nephrology): Prune Belly Syndrome
- `bladder_exstrophy` (Nephrology): Bladder Exstrophy
- `x_linked_adrenoleukodystrophy` (Neurology): X-Linked Adrenoleukodystrophy
- `melkersson_rosenthal_syndrome` (Neurology): Melkersson-Rosenthal Syndrome
- `leber_hereditary_optic_neuropathy` (Neurology): Leber Hereditary Optic Neuropathy
- `primary_congenital_glaucoma` (Ophthalmology): Primary Congenital Glaucoma
- `acute_angle_closure_glaucoma` (Ophthalmology): Acute Angle-Closure Glaucoma
- `trachoma` (Ophthalmology): Trachoma
- `onchocerciasis` (Ophthalmology): Onchocerciasis
- `ocular_coloboma` (Ophthalmology): Ocular Coloboma
- `peters_anomaly` (Ophthalmology): Peters Anomaly
- `sympathetic_ophthalmia` (Ophthalmology): Sympathetic Ophthalmia
- `morning_glory_disc` (Ophthalmology): Morning Glory Disc Anomaly
- `chronic_recurrent_multifocal_osteomyelitis` (Orthopedics): Chronic Recurrent Multifocal Osteomyelitis
- `pyknodysostosis` (Orthopedics): Pyknodysostosis
- `caffey_disease` (Orthopedics): Caffey Disease
- `diastrophic_dysplasia` (Orthopedics): Diastrophic Dysplasia
- `dysplasia_epiphysealis_hemimelica` (Orthopedics): Dysplasia Epiphysealis Hemimelica
- `dupuytren_contracture` (Orthopedics): Dupuytren Contracture
- `fetal_alcohol_spectrum_disorder` (Pediatrics): Fetal Alcohol Spectrum Disorder
- `chromosome_22q11_deletion_syndrome` (Pediatrics): 22q11.2 Deletion Syndrome
- `stickler_syndrome` (Pediatrics): Stickler Syndrome
- `proteus_syndrome` (Pediatrics): Proteus Syndrome
- `recurrent_respiratory_papillomatosis` (Pulmonology): Recurrent Respiratory Papillomatosis
- `diffuse_panbronchiolitis` (Pulmonology): Diffuse Panbronchiolitis
- `pulmonary_arteriovenous_malformation` (Pulmonology): Pulmonary Arteriovenous Malformation
- `deficiency_of_adenosine_deaminase_2` (Rheumatology): Deficiency of Adenosine Deaminase 2
- `blau_syndrome` (Rheumatology): Blau Syndrome
- `schnitzler_syndrome` (Rheumatology): Schnitzler Syndrome
- `hypocomplementemic_urticarial_vasculitis` (Rheumatology): Hypocomplementemic Urticarial Vasculitis

## Not duplicated

Wilson disease remains a single neurology topic. Hepatic copper overload, Kayser-Fleischer rings, and the giant-panda MRI sign are sites of that disease, not a second topic.

## Already present, not added again

- `beta_thalassemia_major`
- `hodgkin_lymphoma`
- `acute_promyelocytic_leukemia`

## Kept on purpose

- Disease-level entities with mainly endoscopic, fundus, or radiographic images stay if that image is how the disease is recognized (achalasia, eosinophilic esophagitis, retinoblastoma, tetralogy of Fallot).
- Dermatitis herpetiformis stays in dermatology and celiac disease stays in gastroenterology because each has its own image set (IgA granules versus duodenal endoscopy), with the overlap noted rather than merged.
- Amebic liver abscess stays beside intestinal amebiasis for the same reason: flask ulcers versus a liver lesion.
- Neurofibromatosis, tuberous sclerosis, and Sturge-Weber stay in neurology rather than being copied into dermatology.
- Adult hypothyroidism and diabetes mellitus were already absent and were not added. Addisonian hyperpigmentation and congenital glaucoma were added because they have photographs.

## Second-pass additions

Added so specialties that lost imaging-only or manifestation topics still finish above their starting count.

- `infantile_pompe_disease` (Cardiology): Infantile-Onset Pompe Disease
- `hypoplastic_left_heart_syndrome` (Cardiology): Hypoplastic Left Heart Syndrome
- `truncus_arteriosus` (Cardiology): Truncus Arteriosus
- `tricuspid_atresia` (Cardiology): Tricuspid Atresia
- `pulmonary_atresia_intact_septum` (Cardiology): Pulmonary Atresia with Intact Ventricular Septum
- `noonan_syndrome_with_multiple_lentigines` (Cardiology): Noonan Syndrome with Multiple Lentigines
- `double_aortic_arch` (Cardiology): Double Aortic Arch
- `aortopulmonary_window` (Cardiology): Aortopulmonary Window
- `barth_syndrome` (Cardiology): Barth Syndrome
- `danon_disease` (Cardiology): Danon Disease
- `exogenous_lipoid_pneumonia` (Pulmonology): Exogenous Lipoid Pneumonia
- `thoracic_endometriosis` (Pulmonology): Thoracic Endometriosis
- `congenital_generalized_lipodystrophy` (Endocrinology): Congenital Generalized Lipodystrophy
- `familial_partial_lipodystrophy` (Endocrinology): Familial Partial Lipodystrophy
- `laron_syndrome` (Endocrinology): Laron Syndrome
- `parathyroid_carcinoma` (Endocrinology): Parathyroid Carcinoma
- `x_linked_hypophosphatemic_rickets` (Endocrinology): X-Linked Hypophosphatemic Rickets
- `juvenile_polyposis_syndrome` (Gastroenterology): Juvenile Polyposis Syndrome
- `howel_evans_syndrome` (Gastroenterology): Howel-Evans Syndrome
- `primary_intestinal_lymphangiectasia` (Gastroenterology): Primary Intestinal Lymphangiectasia
- `eosinophilic_gastroenteritis` (Gastroenterology): Eosinophilic Gastroenteritis
- `solitary_rectal_ulcer_syndrome` (Gastroenterology): Solitary Rectal Ulcer Syndrome
- `pseudomembranous_colitis` (Gastroenterology): Pseudomembranous Colitis
- `acute_graft_versus_host_disease` (Gastroenterology): Acute Graft-versus-Host Disease
- `plummer_vinson_syndrome` (Gastroenterology): Plummer-Vinson Syndrome
- `alagille_syndrome` (Gastroenterology): Alagille Syndrome
- `abetalipoproteinemia` (Gastroenterology): Abetalipoproteinemia
- `branchio_oto_renal_syndrome` (Nephrology): Branchio-Oto-Renal Syndrome
- `lowe_syndrome` (Nephrology): Lowe Syndrome
- `bardet_biedl_syndrome` (Nephrology): Bardet-Biedl Syndrome
- `finnish_congenital_nephrotic_syndrome` (Nephrology): Congenital Nephrotic Syndrome, Finnish Type
- `townes_brocks_syndrome` (Nephrology): Townes-Brocks Syndrome
- `denys_drash_syndrome` (Nephrology): Denys-Drash Syndrome
- `wagr_syndrome` (Nephrology): WAGR Syndrome
- `senior_loken_syndrome` (Nephrology): Senior-Loken Syndrome
- `familial_lcat_deficiency` (Nephrology): Familial LCAT Deficiency
- `cockayne_syndrome` (Neurology): Cockayne Syndrome
- `niemann_pick_type_c` (Neurology): Niemann-Pick Disease Type C
- `lgi1_antibody_encephalitis` (Neurology): LGI1-Antibody Encephalitis
- `canavan_disease` (Neurology): Canavan Disease
- `zellweger_syndrome` (Neurology): Zellweger Syndrome
- `yellow_nail_syndrome` (Pulmonology): Yellow Nail Syndrome
- `chronic_eosinophilic_pneumonia` (Pulmonology): Chronic Eosinophilic Pneumonia
- `diffuse_pulmonary_lymphangiomatosis` (Pulmonology): Diffuse Pulmonary Lymphangiomatosis
- `aspirin_exacerbated_respiratory_disease` (Pulmonology): Aspirin-Exacerbated Respiratory Disease
- `pulmonary_hyalinizing_granuloma` (Pulmonology): Pulmonary Hyalinizing Granuloma
- `acute_eosinophilic_pneumonia` (Pulmonology): Acute Eosinophilic Pneumonia
- `blue_rubber_bleb_nevus_syndrome` (Dermatology): Blue Rubber Bleb Nevus Syndrome
- `acrodermatitis_enteropathica` (Dermatology): Acrodermatitis Enteropathica
- `pellagra` (Dermatology): Pellagra
- `merkel_cell_carcinoma` (Dermatology): Merkel Cell Carcinoma
- `cutaneous_lupus_erythematosus` (Dermatology): Cutaneous Lupus Erythematosus

Active counts after both passes: Cardiology 35, Dermatology 63, Endocrinology 25, Gastroenterology 39, Hematology 29, Infectious Diseases 52, Nephrology 25, Neurology 45, Ophthalmology 26, Orthopedics 46, Pediatrics 38, Pulmonology 29, Rheumatology 43.
