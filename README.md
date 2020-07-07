# Spotify-Classification-Regression-Challenge
- This project involved an analysis of Spotify songs as part of a Kaggle competition.
- A dataset that captured the various attributes about songs including their genre and popularity score was utilized to address two problems: 
  - A regression problem which aims to predict the popularity score of a song.
    - For this problem I utilized a Random Forest Regressor, hyper-tuned extensively achieve an RMSE score of 0.545589.
  - A classification problem which aims to predict the top genre that a song belongs to
    - For this problem I fit two classifiers, Adaptive Boost and Random Forest in tandem with the techniques for balancing heavily imbalanced data called Random Over Sampling and Synthetic Minority Over Sampling Technique. The adaptive boosted classifier produced the highest score, with an accuracy, precision, and recall score of 0.98.
