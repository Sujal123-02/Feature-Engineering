# Feature-Engineering





The preprocessing methodology for the **Sign Language MNIST** and **SMS Spam Collection** datasets focused on preparing their unique data formats for machine learning classifiers. For the image-based Sign Language MNIST, the main tasks involved handling the high-dimensionality and specific data scale: features were already numerical (pixel values), so the essential steps were **Normalization** (scaling all pixel values to the $0.0-1.0$ range for stability) and **Dimensionality Reduction** using **Principal Component Analysis (PCA)** to efficiently compress the $784$ pixel features, making the **Random Forest** model faster to train. Since the data was clean, imputation and encoding were skipped.

Conversely, the **SMS Spam Collection** dataset, which used engineered tabular features derived from text, required a broader range of transformations to ensure quality and compatibility. This involved using **Mean Imputation** to fill missing numerical values (like `NumURLs`) and **One-Hot Encoding** to convert categorical features (`Sentiment_Class`) into a numerical format. Subsequently, numerical features underwent **Standardization** ($\mu=0, \sigma=1$) to prevent features like `MessageLength` from dominating the model, followed by **Feature Selection** via $\text{SelectKBest}$ to isolate the most predictive variables for the 'spam'/'ham' classification. Both pipelines successfully converted raw data into robust input matrices, demonstrating compliance with all assignment guidelines.
