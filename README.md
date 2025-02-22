# pro-disorder-predictor
Predicting Disordered Regions in Proteins Using Sequence-Based Features and Structural Insights
Intrinsically disordered regions (IDRs) in proteins play a crucial role in various biological functions, including molecular recognition, signaling, and regulation. Unlike structured regions, IDRs lack a fixed three-dimensional conformation, making their prediction challenging but essential for understanding protein function. Identifying disordered regions helps in structural biology, drug discovery, and disease research, particularly in neurodegenerative disorders and cancer. In this project, we explored computational approaches to predict IDRs using sequence-based features and structural data.

Our MLP-based model outperformed IUPred and was competitive with DISOPRED2. While DISOPRED3 remains a strong benchmark, our approach demonstrated promising results by leveraging ESM2 embeddings and structural features from AlphaFold.

Model Comparison

Our vs Existing

Model

Balanced Accuracy

AUC

Ours

MLP

0.79

0.84



Logistic Regression

0.74

0.82



HMM

0.73

0.79



AF confidence

0.72

0.75

Existing

IUPred

0.78

0.76



DISOPRED2

0.62

0.74



DISOPRED3

0.79

0.90

Next Steps

Looking forward, several improvements could be made:

Incorporate Additional Structural Data: Enhancing feature extraction using full 3D structure predictions from AlphaFold could improve accuracy.

Refine Training with More Data: Expanding the dataset to include additional IDP annotations from external databases would increase robustness.

Hybrid Model Approaches: Combining our MLP with HMM or transformer-based models could yield better predictions.

Post-processing for Interpretability: Developing visualization tools to highlight confidence scores and disorder regions in protein structures would aid in biological understanding.

Benchmarking Against More Datasets: Evaluating performance in independent test sets from different sources would validate generalizability.

These extensions would improve prediction accuracy and biological relevance, making our tool more useful to the research community.

We think that more data processing may enable us to utilize the pLDDT score for better predictions.

