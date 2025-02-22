# pro-disorder-predictor
## Predicting Disordered Regions in Proteins Using Sequence-Based Features and Structural Insights

Intrinsically disordered regions (IDRs) in proteins play a crucial role in various biological functions, including molecular recognition, signaling, and regulation. Unlike structured regions, IDRs lack a fixed three-dimensional conformation, making their prediction challenging but essential for understanding protein function. Identifying disordered regions helps in structural biology, drug discovery, and disease research, particularly in neurodegenerative disorders and cancer. In this project, we explored computational approaches to predict IDRs using sequence-based features and structural data.

Our MLP-based model outperformed IUPred and was competitive with DISOPRED2. While DISOPRED3 remains a strong benchmark, our approach demonstrated promising results by leveraging ESM2 embeddings and structural features from AlphaFold.

## Model Comparison

| **Our vs Existing** | **Model**                 | **Balanced Accuracy** | **AUC** |
|---------------------|--------------------------|----------------------|---------|
| **Ours**           | **MLP**                   | **0.79**             | **0.84** |
|                     | Logistic Regression      | 0.74                 | 0.82    |
|                     | HMM                      | 0.73                 | 0.79    |
|                     | AF confidence            | 0.72                 | 0.75    |
| **Existing**        | IUPred                   | 0.78                 | 0.76    |
|                     | DISOPRED2                | 0.62                 | 0.74    |
|                     | **DISOPRED3**            | **0.79**             | **0.90** |

## Next Steps

Looking forward, several improvements could be made:

1. **Incorporate Additional Structural Data**: Enhancing feature extraction using full 3D structure predictions from AlphaFold could improve accuracy.
2. **Refine Training with More Data**: Expanding the dataset to include additional IDP annotations from external databases would increase robustness.
3. **Hybrid Model Approaches**: Combining our MLP with HMM or transformer-based models could yield better predictions.
4. **Post-processing for Interpretability**: Developing visualization tools to highlight confidence scores and disorder regions in protein structures would aid in biological understanding.
5. **Benchmarking Against More Datasets**: Evaluating performance in independent test sets from different sources would validate generalizability.

These extensions would improve prediction accuracy and biological relevance, making our tool more useful to the research community.
We think that more data processing may enable us to utilize the pLDDT score for better predictions.

## References
- [Aspromonte et al., 2024](https://academic.oup.com/nar/article/52/D1/D434/7334088) - DisProt in 2024: improving function annotation of intrinsically disordered proteins.
- [Lin et al., 2022](https://doi.org/10.1101/2022.07.20.500902) - Evolutionary-scale prediction of atomic level protein structure with a language model.
- [Jumper et al., 2021](https://doi.org/10.1038/s41586-021-03819-2) - Highly accurate protein structure prediction with AlphaFold.
- [Varadi et al., 2022](https://doi.org/10.1093/nar/gkab1061) - AlphaFold Protein Structure Database: massively expanding the structural coverage of protein-sequence space with high-accuracy models.
- [Dosztányi et al., 2005](https://doi.org/10.1093/bioinformatics/bti541) - IUPred: web server for the prediction of intrinsically unstructured regions of proteins.
- [Jones & Cozzetto, 2015](https://doi.org/10.1093/bioinformatics/btu744) - DISOPRED3: precise disordered region predictions with annotated protein-binding activity.

