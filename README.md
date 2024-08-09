<h1>Predicting Evaly's Profit with Python</h1>

<p>This repository contains the code for predicting Evaly's profit using a Long Short-Term Memory (LSTM) neural network. The project is implemented in a Jupyter notebook and executed in your local environment or any preferred platform that supports Jupyter notebooks.</p>

<h2>Table of Contents</h2>
<ul>
    <li><a href="#introduction">Introduction</a></li>
    <li><a href="#data-acquisition-and-preprocessing">Data Acquisition and Preprocessing</a></li>
    <li><a href="#model-selection-and-training">Model Selection and Training</a></li>
    <li><a href="#evaluation-and-prediction">Evaluation and Prediction</a></li>
    <li><a href="#visualization">Visualization</a></li>
    <li><a href="#limitations-of-the-model">Limitations of the Model</a></li>
    <li><a href="#potential-improvements">Potential Improvements</a></li>
    <li><a href="#how-to-run-the-code">How to Run the Code</a></li>
</ul>

<h2 id="introduction">Introduction</h2>
<p>This project aims to predict Evaly's future profit by analyzing historical campaign product data, including details like product names, offer prices, and regular prices. The model used for this purpose is a Long Short-Term Memory (LSTM) neural network, known for its effectiveness in time series forecasting.</p>

<h2 id="data-acquisition-and-preprocessing">Data Acquisition and Preprocessing</h2>
<p>The dataset used in this project includes Evaly's campaign product data. The data is loaded into a pandas DataFrame, where initial exploratory data analysis (EDA) is performed. The data is then preprocessed by handling missing values, creating lag features, and calculating statistical features such as rolling means and standard deviations to enrich the dataset.</p>

<h2 id="model-selection-and-training">Model Selection and Training</h2>
<p>A Long Short-Term Memory (LSTM) neural network is selected due to its ability to handle sequential dependencies in time series data. The LSTM model is constructed using the <code>Sequential</code> API from <code>Keras</code>, with layers added to capture the necessary patterns. The model is compiled using the Adam optimizer and mean squared error loss function. Training is conducted with early stopping to prevent overfitting, and the best model is saved based on validation loss.</p>

<h2 id="evaluation-and-prediction">Evaluation and Prediction</h2>
<p>The trained LSTM model is evaluated on both training and testing datasets using metrics such as Mean Absolute Error (MAE) and Mean Squared Error (MSE). Predictions are made for future profit, and the predicted values are transformed back to their original scale for comparison with actual values.</p>

<h2 id="visualization">Visualization</h2>
<p>Visualizations are created to compare actual vs. predicted profits. Plots are generated to show the performance of the model during both the training and testing phases, with clear annotations to help interpret the results.</p>

<h2 id="limitations-of-the-model">Limitations of the Model</h2>
<ul>
    <li><strong>Limited Feature Engineering:</strong> The model relies on basic features derived from the dataset. Additional features could improve performance.</li>
    <li><strong>Assumptions in Data:</strong> The model assumes a consistent growth rate and does not account for external factors that could impact profits.</li>
</ul>

<h2 id="potential-improvements">Potential Improvements</h2>
<ul>
    <li><strong>Advanced Feature Engineering:</strong> Incorporating more complex features such as external economic factors, competitor data, or seasonal trends could improve accuracy.</li>
    <li><strong>Hyperparameter Tuning:</strong> More extensive hyperparameter tuning could help in optimizing the model's performance.</li>
    <li><strong>Data Augmentation:</strong> Expanding the dataset with additional relevant data and ensuring data quality can further enhance the model's predictions.</li>
</ul>

<h2 id="how-to-run-the-code">How to Run the Code</h2>
<p>To run the code, follow these steps:</p>
<ol>
    <li><strong>Clone the Repository:</strong>
        <pre><code>git clone https://github.com/yourusername/evaly-profit-prediction.git
cd evaly-profit-prediction
</code></pre>
    </li>
    <li><strong>Open the Notebook:</strong>
        <ul>
            <li>Open the Jupyter notebook (<code>Predicting_Evaly’s_profit_with_Python_Using_LSTM_Model_ByTanjela.ipynb</code>) in your preferred environment.</li>
        </ul>
    </li>
    <li><strong>Install Required Libraries:</strong>
        <p>Install the necessary libraries by running:</p>
        <pre><code>!pip install pandas numpy matplotlib seaborn scikit-learn tensorflow</code></pre>
    </li>
    <li><strong>Run the Notebook:</strong>
        <p>Execute each cell in the notebook sequentially, following any instructions for specific configurations or data preprocessing steps.</p>
    </li>
</ol>
<p>By following these steps, anyone can replicate the process and run the code to predict Evaly's future profit.</p>

<h2 id="screenshot-of-code-running">Screenshot of Code Running</h2>
<p><img src="path_to_screenshot" alt="![image](https://github.com/user-attachments/assets/ce2f0a81-2ede-4d64-9ca3-7c0dc0d9c6c5)
"></p>
