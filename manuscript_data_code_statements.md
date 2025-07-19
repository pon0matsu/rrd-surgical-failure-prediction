# 原稿に記載可能なData and Code Availabilityの例文

## 1. Methods section内での記載例

### Model availabilityの項（現在の記載を補強）:
```
We developed a web-based application to facilitate real-time risk assessment for surgical failure in rhegmatogenous retinal detachment cases. The tool implements our TabPFN classification model and features a user-friendly interface for inputting patient variables. The application calculates a failure risk score ≥ 0.7 for high-risk classification. The application is available through Hugging Face Spaces (https://huggingface.co/spaces/JRVS-DSDP/ppv-rrd-risk-stratification, accessed on May 20th, 2025). The model is based on the J-RD Registry dataset and validated with Wills Eye Hospital data. The development data can be accessed through the J-RD Registry application process, while the validation dataset is not publicly available due to privacy restrictions. Code and documentation are available at https://github.com/[your-username]/rrd-surgical-failure-prediction.
```

## 2. 論文末尾のData Availability Statementの例

### オプション1（標準的な記載）:
```
Data Availability Statement
The data underlying this study are available from the Japan-Retinal Detachment Registry upon reasonable request through the Japan Retina and Vitreous Society (https://www.jrvs.jp/works/). The external validation dataset from Wills Eye Hospital cannot be made publicly available due to patient privacy regulations and institutional policies. The prediction model is publicly accessible through a web application at https://huggingface.co/spaces/JRVS-DSDP/ppv-rrd-risk-stratification. Code for model development and evaluation will be made available at https://github.com/[your-username]/rrd-surgical-failure-prediction upon publication.
```

### オプション2（より詳細な記載）:
```
Data and Code Availability
Development dataset: The Japan-Retinal Detachment (J-RD) Registry data supporting this study's findings are available through formal application to the Japan Retina and Vitreous Society at https://www.jrvs.jp/works/. Access requires institutional approval and execution of a data use agreement.

External validation dataset: The Wills Eye Hospital dataset cannot be shared publicly due to patient privacy restrictions under HIPAA regulations. Researchers interested in collaboration may contact the corresponding authors.

Code availability: The trained prediction model is deployed and freely accessible via https://huggingface.co/spaces/JRVS-DSDP/ppv-rrd-risk-stratification. Source code for data preprocessing, model training, and evaluation procedures is available from the corresponding authors upon reasonable request. A public repository with documentation is maintained at https://github.com/[your-username]/rrd-surgical-failure-prediction.
```

## 3. TRIPODチェックリスト用の記載

### Item 21 (Supplementary information):
```
Supplementary information is available at the journal's website and includes detailed feature descriptions (Table S1-S6), SHAP analysis results (Figure S1), and the complete TRIPOD checklist.
```

### Item 22 (Model and data availability):
```
The final prediction model is available as a web application (https://huggingface.co/spaces/JRVS-DSDP/ppv-rrd-risk-stratification). Development data can be accessed through J-RD Registry application. External validation data are not publicly available due to privacy restrictions. Code is available from the corresponding authors and partially documented at https://github.com/[your-username]/rrd-surgical-failure-prediction.
```

## 4. Cover Letter用の記載例:
```
To ensure transparency and reproducibility, we have made our prediction model publicly available through a web interface on HuggingFace Spaces. While the primary dataset requires formal application through the Japan Retina and Vitreous Society, we have provided comprehensive documentation of our methods and made our code available upon request. This approach balances open science principles with necessary patient privacy protections.
```

## 注意事項:
- [your-username]は実際のGitHubユーザー名に置き換えてください
- ジャーナルによってはData Availability Statementの形式が指定されている場合があるので、投稿規定を確認してください
- 一部のジャーナルでは、"Code available upon request"よりも完全公開が推奨される場合があります