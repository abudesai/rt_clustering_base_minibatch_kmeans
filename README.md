K-Means clustering in Sklearn for Clustering - Base problem category as per Ready Tensor specifications.

- k-means
- clustering
- sklearn
- python
- pandas
- numpy
- flask
- nginx
- uvicorn
- docker

This is a Clustering Model that uses k-means implemented through Sklearn.

The algorithm aims to partition n observations into k clusters in which each observation belongs to the cluster with the nearest mean (cluster centers or cluster centroid), serving as a prototype of the cluster.

The data preprocessing step includes:

- for numerical variables
  - Standard scale data

During the model development process, the algorithm was trained and evaluated on a variety of datasets such as iris, penguins, landsat_satellite, geture_phase_classification, vehicle_silhouettes, spambase, steel_plate_fault. Additionally, we also used synthetically generated datasets such as two concentric (noisy) circles, and unequal variance gaussian blobs.

This Clustering Model is written using Python as its programming language. ScikitLearn is used to implement the main algorithm, evaluate the model, and preprocess the data. Numpy, pandas, Sklearn, and feature-engine are used for the data preprocessing steps.

The model includes an inference service with 2 endpoints: /ping for health check and /infer for predictions of nearest clusters in real time. The inference service is implemented using flask+nginx+uvicorn.
