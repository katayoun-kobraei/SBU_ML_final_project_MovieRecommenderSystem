# Movie Recommender System

## Overview
The **Movie Recommender System** project aims to provide personalized movie suggestions based on user preferences using machine learning techniques. Users can input three movies they like, and the system will recommend similar films based on a comprehensive dataset.


## Project Components

### 1. Dataset
- **Source**: The dataset consists of metadata for 45,000 movies sourced from the **The Movie Database (TMDb) API** and includes:
  - **Attributes**: Titles, genres, cast, crew, budgets, revenues, release dates, languages, and user ratings.
  - **User Ratings**: Contains ratings from 270,000 users, facilitating collaborative filtering.

### 2. Model
- **Technique**: The system employs **collaborative filtering**, specifically using cosine similarity to find and recommend movies that are similar to those the user has already liked.
- **Functionality**:
  - When a user inputs three movies, the model calculates pairwise distances between the chosen movies and all other movies in the dataset.
  - It ranks movies based on similarity scores and generates a list of recommended films.

### 3. Data Preprocessing
- **Steps Involved**:
  - **Loading Data**: Reading CSV files containing movie metadata.
  - **Cleaning**: Handling missing values, removing duplicates, and converting data types.
  - **Feature Engineering**: Creating relevant features from the dataset to enhance similarity calculations.
  - **Vectorization**: Transforming categorical data into numerical formats using techniques like CountVectorizer.

### 4. User Interface
- **Framework**: Built using **Streamlit**, which allows for rapid development of web applications in Python.
- **Functionality**:
  - Users can input their favorite movies and ratings through an intuitive interface.
  - Recommended movies are displayed alongside their details, including titles, release years, genres, and posters.

### 5. Evaluation
- **Metrics**: The model’s performance is assessed using:
  - **Precision**: The proportion of relevant recommendations among all recommended movies.
  - **Recall**: The proportion of relevant movies that were recommended out of all relevant movies available.
  - **F1 Score**: The harmonic mean of precision and recall, providing a single measure of the model's accuracy.

### 6. Deployment
- **Platform**: The application is deployed on **Hugging Face Spaces**, making it easily accessible online.
- **Process**: Involves creating a space, pushing code, and setting up the environment to run the Streamlit application.

## Usage
1. **Input**: Users enter three movies they have watched and enjoyed.
2. **Recommendation**: The system processes the input and provides a list of similar movies.
3. **Interaction**: Users can view additional movie information and explore recommendations.

## Access
To explore the deployed application, visit: [Movie Recommender System](https://huggingface.co/spaces/Mehrdadbn/Movie-recommender-system)

## Conclusion
In this project, we combined machine learning techniques with a user-friendly interface to create a functional movie recommendation system. By leveraging collaborative filtering and extensive movie data, the model provides personalized suggestions that enhance user experience in discovering new films.
