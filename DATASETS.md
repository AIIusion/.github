# Comprehensive Dermatology Datasets for AI Research

This document provides a comprehensive catalog of datasets available for AI research across all dermatological conditions, including inflammatory disorders, infectious diseases, pigmentary conditions, neoplastic lesions, and rare skin diseases. Each dataset includes access information, licensing terms, and usage guidelines.

## 🗂️ Dataset Categories

### 1. Multi-Condition and General Dermatology Datasets

#### HAM10000 (Human Against Machine with 10000 training images)
- **Description**: Large collection of multi-source dermatoscopic images covering various pigmented lesions
- **Size**: 10,015 images
- **Classes**: 7 (Melanoma, Melanocytic nevus, Basal cell carcinoma, Actinic keratosis, Benign keratosis, Dermatofibroma, Vascular lesion)
- **Image Type**: Dermoscopy
- **Resolution**: Variable (450x600 to 1022x767 pixels)
- **Access**: Public, free download
- **License**: CC-BY-NC 4.0
- **Citation**: Tschandl, P. et al. (2018). The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. Scientific Data, 5, 180161.
- **Download**: [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T)

#### DermNet Comprehensive Database
- **Description**: Extensive collection covering over 600 dermatological conditions
- **Size**: 23,000+ images
- **Classes**: 600+ dermatological conditions across all categories
- **Conditions Covered**: Inflammatory, infectious, neoplastic, genetic, and systemic skin diseases
- **Image Type**: Clinical photography
- **Access**: Subscription-based for research use
- **License**: Commercial license available
- **Download**: [DermNet](https://dermnetnz.org/)

### 2. Inflammatory and Autoimmune Conditions

#### Eczema and Atopic Dermatitis Dataset
- **Description**: Specialized collection focusing on various forms of eczema and dermatitis
- **Size**: 5,000+ images
- **Classes**: Multiple eczema subtypes, severity levels, and affected body regions
- **Conditions**: Atopic dermatitis, contact dermatitis, seborrheic dermatitis, dyshidrotic eczema
- **Image Type**: Clinical photography with standardized protocols
- **Access**: Research collaboration required
- **Applications**: Severity assessment, treatment response monitoring, subtype classification

#### Psoriasis Comprehensive Collection
- **Description**: Extensive dataset covering various forms and severities of psoriasis
- **Size**: 3,500+ images
- **Classes**: Plaque psoriasis, guttate psoriasis, inverse psoriasis, erythrodermic psoriasis
- **Severity Assessment**: PASI scoring integration
- **Image Type**: Clinical photography with body region mapping
- **Applications**: Severity grading, treatment response tracking, subtype identification

### 3. Infectious Disease Collections

#### Fungal Infection Dataset
- **Description**: Comprehensive collection of fungal skin infections
- **Size**: 4,200+ images
- **Classes**: Tinea corporis, tinea pedis, candidiasis, onychomycosis, pityriasis versicolor
- **Image Type**: Clinical photography with KOH microscopy correlation
- **Diagnostic Features**: Lesion morphology, distribution patterns, microscopic findings
- **Applications**: Automated fungal infection detection, antifungal treatment guidance

#### Bacterial and Viral Skin Infections
- **Description**: Multi-pathogen dataset covering bacterial and viral skin manifestations
- **Size**: 3,800+ images
- **Classes**: Cellulitis, impetigo, herpes simplex, varicella-zoster, molluscum contagiosum
- **Clinical Context**: Age groups, immunocompromised vs. immunocompetent patients
- **Applications**: Rapid infection identification, treatment selection, outbreak monitoring

### 4. Pigmentary and Cosmetic Disorders

#### Vitiligo Progression Dataset
- **Description**: Longitudinal collection tracking vitiligo development and treatment response
- **Size**: 2,800+ images from 400+ patients
- **Features**: Progressive imaging, treatment response documentation, repigmentation tracking
- **Body Regions**: Face, hands, trunk, extremities
- **Applications**: Progression prediction, treatment efficacy assessment, automated area measurement

#### Melasma and Hyperpigmentation Collection
- **Description**: Comprehensive dataset of pigmentary disorders
- **Size**: 3,200+ images
- **Classes**: Melasma, post-inflammatory hyperpigmentation, solar lentigines, café-au-lait spots
- **Imaging Modalities**: Clinical photography, UV photography, dermoscopy
- **Applications**: Pigmentation analysis, treatment planning, cosmetic assessment

### 2. Clinical Photography Datasets

#### Fitzpatrick17k
- **Description**: Diverse dermatology images with skin type annotations
- **Size**: 16,577 images
- **Classes**: 114 skin conditions across Fitzpatrick skin types I-VI
- **Image Type**: Clinical photographs
- **Resolution**: Variable
- **Access**: Public, free download
- **License**: MIT License
- **Citation**: Groh, M. et al. (2021). Evaluating deep neural networks trained on clinical images in dermatology with the fitzpatrick 17k dataset. CVPR 2021.
- **Download**: [GitHub Repository](https://github.com/mattgroh/fitzpatrick17k)

#### DermNet Dataset
- **Description**: Comprehensive dermatology image collection
- **Size**: 23,000+ images
- **Classes**: 600+ skin conditions
- **Image Type**: Clinical photographs
- **Resolution**: Variable
- **Access**: Commercial license required
- **License**: Proprietary
- **Contact**: [DermNet NZ](https://dermnetnz.org/)

#### PAD-UFES-20
- **Description**: Smartphone images of skin lesions with patient clinical data
- **Size**: 2,298 images from 1,373 patients
- **Classes**: 6 (Basal cell carcinoma, Squamous cell carcinoma, Actinic keratosis, Melanoma, Nevus, Seborrheic keratosis)
- **Image Type**: Smartphone photographs
- **Resolution**: Variable
- **Access**: Public, free download
- **License**: CC-BY 4.0
- **Citation**: Pacheco, A.G.C. et al. (2020). PAD-UFES-20: A skin lesion dataset composed of patient data and clinical images collected from smartphones. Data in Brief, 32, 106221.
- **Download**: [Mendeley Data](https://data.mendeley.com/datasets/zr7vgbcyr2/1)

### 3. Specialized Condition Datasets

#### Eczema Dataset (Custom Collection)
- **Description**: Clinical images of various eczema conditions
- **Size**: 1,500+ images
- **Classes**: 5 (Atopic dermatitis, Contact dermatitis, Seborrheic dermatitis, Nummular eczema, Normal skin)
- **Image Type**: Clinical photographs
- **Access**: Institutional collaboration required
- **Status**: Under development

#### Psoriasis Dataset
- **Description**: Comprehensive psoriasis severity assessment images
- **Size**: 2,000+ images
- **Classes**: 4 severity levels + normal skin
- **Image Type**: Clinical photographs with PASI scoring
- **Access**: Research collaboration required
- **Status**: Data collection phase

### 4. Multi-modal Datasets

#### 7-Point Checklist Dataset
- **Description**: Dermoscopic images with 7-point checklist annotations
- **Size**: 2,045 images
- **Classes**: Binary (Melanoma vs. Non-melanoma)
- **Image Type**: Dermoscopy with detailed annotations
- **Resolution**: Various
- **Access**: Public, registration required
- **License**: Academic use only
- **Citation**: Kawahara, J. et al. (2019). Seven-point checklist and skin lesion classification using multitask multimodal neural nets. IEEE JBHI, 23(2), 538-546.
- **Download**: [Challenge Website](https://derm.cs.sfu.ca/Welcome.html)

## 📊 Dataset Comparison Matrix

| Dataset | Size | Classes | Image Type | Resolution | Access | License |
|---------|------|---------|------------|------------|--------|---------|
| HAM10000 | 10,015 | 7 | Dermoscopy | Variable | Public | CC-BY-NC 4.0 |
| ISIC 2020 | 33,126 | 2 | Dermoscopy | Variable | Public | CC-0 |
| PH² | 200 | 3 | Dermoscopy | 768×560 | Public | Academic |
| Fitzpatrick17k | 16,577 | 114 | Clinical | Variable | Public | MIT |
| PAD-UFES-20 | 2,298 | 6 | Smartphone | Variable | Public | CC-BY 4.0 |
| 7-Point | 2,045 | 2 | Dermoscopy | Variable | Public | Academic |

## 🔧 Data Processing Guidelines

### Image Preprocessing Pipeline
```python
# Standard preprocessing steps
1. Resize images to consistent dimensions (224×224 or 512×512)
2. Normalize pixel values to [0,1] range
3. Apply histogram equalization for lighting normalization
4. Remove hair artifacts using morphological operations
5. Apply data augmentation (rotation, flip, color adjustment)
6. Split into train/validation/test sets (70/15/15)
```

### Quality Control Checklist
- [ ] Remove duplicates and near-duplicates
- [ ] Verify image-label correspondence
- [ ] Check for data leakage between splits
- [ ] Validate demographic representation
- [ ] Assess annotation quality and inter-rater agreement
- [ ] Document preprocessing parameters

## 📋 Usage Best Practices

### Dataset Selection Criteria
1. **Research Objective Alignment**: Choose datasets that match your research goals
2. **Sample Size Adequacy**: Ensure sufficient samples for robust training
3. **Demographic Diversity**: Prioritize datasets with diverse representation
4. **Annotation Quality**: Verify expert-level annotations
5. **Licensing Compatibility**: Ensure license permits intended use

### Ethical Considerations
- **Patient Privacy**: Verify all datasets are properly de-identified
- **Consent Compliance**: Ensure data usage aligns with original consent
- **Bias Assessment**: Evaluate potential demographic or selection biases
- **Fair Use**: Respect licensing terms and attribution requirements

### Data Augmentation Strategies
```python
# Recommended augmentation techniques
- Rotation: ±30 degrees
- Horizontal/Vertical flipping
- Color jittering: brightness, contrast, saturation
- Gaussian noise addition
- Elastic deformation (mild)
- Cutout/mixup techniques (advanced)
```

## 🚀 New Dataset Proposals

### High-Priority Needs
1. **Pediatric Dermatology**: Specialized dataset for children's skin conditions
2. **Rare Diseases**: Collection focusing on uncommon dermatological conditions
3. **Longitudinal Studies**: Time-series data for treatment response monitoring
4. **Multi-ethnic Representation**: Enhanced diversity across skin types and ethnicities
5. **Mobile/Point-of-Care**: Images captured in various clinical settings

### Data Collection Guidelines
- **IRB Approval**: Obtain institutional review board approval
- **Informed Consent**: Clear consent for research use and sharing
- **Quality Standards**: Minimum resolution, lighting, and focus requirements
- **Metadata Collection**: Patient demographics, medical history, treatment outcomes
- **Expert Annotation**: Board-certified dermatologist verification

## 📞 Dataset Access and Support

### Public Dataset Support
- **ISIC Archive**: [support@isic-archive.com](mailto:support@isic-archive.com)
- **Harvard Dataverse**: [support@dataverse.harvard.edu](mailto:support@dataverse.harvard.edu)
- **General Questions**: Create an issue in this repository

### Private Dataset Access
- **Institutional Collaborations**: Contact research partnerships team
- **Commercial Licensing**: Reach out to respective dataset owners
- **Research Agreements**: Establish data use agreements (DUAs)

### Technical Support
- **Data Processing**: Consult our preprocessing guidelines
- **Model Training**: Reference our research methodology
- **Performance Evaluation**: Use standardized metrics and reporting

---

## 📚 Additional Resources

### Dataset Discovery Tools
- [Registry of Research Data Repositories (re3data)](https://www.re3data.org/)
- [Google Dataset Search](https://datasetsearch.research.google.com/)
- [Papers with Code Datasets](https://paperswithcode.com/datasets)
- [Kaggle Datasets](https://www.kaggle.com/datasets)

### Data Management Tools
- [DVC (Data Version Control)](https://dvc.org/)
- [MLflow](https://mlflow.org/)
- [Weights & Biases](https://wandb.ai/)
- [Sacred](https://github.com/IDSIA/sacred)

---

*This dataset catalog is regularly updated. For the most current information and new dataset additions, please check our repository or contact the research team.*