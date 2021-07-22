# ##########################################################
# # HarvardX Capstone Project - Movie Recommender System
# ##########################################################


# ##########################################################
# # Create edx set, validation set (final hold-out test set)
# # (this first section of code provided by HarvardX)
# ##########################################################

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(caret)
library(data.table)
library(dplyr)
library(ggplot2)
library(lubridate)
library(tidyverse)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding")
test_mndex <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_mndex,]
temp <- movielens[test_mndex,]

# Ensure userId and movieId in validation set are also in edx set
validation <- temp %>%
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_mndex, temp, movielens, removed)

library(stringr)
patterns <- c("Drama","Comedy","Thriller","Romance")
edx$genres %>% str_detect("Drama") %>% sum()
sapply(patterns, function(x){
  edx$genres %>% str_detect(x)  %>% sum()
})


# #####################################################
# # Useful functions
# #####################################################
# compute root mean squared error

RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}


# #####################################################
# # create train and test sets from edx
# #####################################################

set.seed(2021)

test_mndex <- createDataPartition(edx$userId, times = 1, p = 0.1, list = FALSE)

test_set <- edx[test_mndex, ]

train_set <- edx[-test_mndex, ]

train_set <- train_set %>% mutate(quarter = round_date(as_datetime(timestamp), 
  "quarter")) # add column for quarter


# #####################################################
# # effects calculations prior to regularization 
# #####################################################

initial_default_pentalty <- 3 # set an initial default penalty of 3
lambda_m <- initial_default_pentalty # init. set regularization penalty parameter to 3 
lambda_g <- initial_default_pentalty # init. set regularization penalty parameter to 3
lambda_u <- initial_default_pentalty # init. set regularization penalty parameter to 3

mu <- mean(train_set$rating)

# compute initial movie, user, genre, and time effects 
# (based on initial lambdas, prior to regularization)

# compute movie effects
movie_avgs <- train_set %>%
  group_by(movieId) %>%
  summarize(b_m = sum(rating - mu)/(n()+lambda_m))

# compute user effects
user_avgs <- train_set %>%
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - mu - b_m)/(n() + lambda_u))

# compute genre effects
genre_avgs <- train_set %>% left_join(movie_avgs,by = "movieId") %>%
  left_join(user_avgs, by = 'userId') %>%
  group_by(genres)%>% summarize(b_g = sum(rating - mu - b_m - b_u)/(n() + lambda_g))

# compute time effects (by quarter, excluding 1995-01-01 due to small # ratings)
quarter_avgs <- train_set %>% filter(quarter != as_date("1995-01-01")) %>% 
  left_join(movie_avgs,by = "movieId") %>% 
  left_join(user_avgs, by = 'userId') %>% 
  left_join(genre_avgs, by = 'genres')%>%
  group_by(quarter) %>% summarize(b_q = mean(rating - mu - b_m - b_u - b_g)) # no regularization for quarter


# #####################################################
# # regularization 
# #####################################################

lambdas <- seq(0,20,1) # perform calcs for range of reg. penalties

# regularization for userId
rmses_u <- sapply(lambdas, function(l){
  # compute movie effects
  movie_avgs <- train_set %>%
  group_by(movieId) %>%
  summarize(b_m = sum(rating - mu)/(n()+l))
  # predict ratings
  predicted_ratings <- test_set %>%
    left_join(movie_avgs, by='movieId') %>%
    left_join(user_avgs, by='userId') %>% left_join(genre_avgs, by = 'genres') %>%
    mutate(pred = mu + b_m + b_u + b_g) %>%
    pull(pred)
  predicted_ratings[is.na(predicted_ratings)] <- mu # assign ave rating to NA
   return(RMSE(predicted_ratings, test_set$rating))
})

lambda_u_optimal <- lambdas[which.min(rmses_u)] # select optimal penalty

# regularization for movieId
rmses_m <- sapply(lambdas, function(l){
  # compute movie effects
  mu <- mean(train_set$rating)
  movie_avgs <- train_set %>%
    group_by(movieId) %>%
    summarize(b_m = sum(rating - mu)/(n()+l))
   # compute predictors
  predicted_ratings <- test_set %>%
    left_join(movie_avgs, by='movieId') %>%
    left_join(user_avgs, by='userId') %>% left_join(genre_avgs, by = 'genres') %>%
    mutate(pred = mu + b_m + b_u + b_g) %>%
    pull(pred)
  predicted_ratings[is.na(predicted_ratings)] <- mu # assign ave rating to NA
  return(RMSE(predicted_ratings, test_set$rating)) 
  })

lambda_m_optimal <- lambdas[which.min(rmses_m)] # select optimal penalty

# regularization for genre effects
rmses_g <- sapply(lambdas, function(l){
   # compute genre effects
  genre_avgs <- train_set %>% left_join(movie_avgs,by = "movieId") %>%
    left_join(user_avgs, by = 'userId') %>% 
    group_by(genres)%>% summarize(b_g = sum(rating - mu - b_m - b_u)/(n() + l))
    # compute predictors
  predicted_ratings <- test_set %>%
    left_join(movie_avgs, by='movieId') %>%
    left_join(user_avgs, by='userId') %>% left_join(genre_avgs, by = 'genres') %>%
    mutate(pred = mu + b_m + b_u + b_g) %>%
    pull(pred)
  predicted_ratings[is.na(predicted_ratings)] <- mu # assign ave rating to NA
  return(RMSE(predicted_ratings, test_set$rating))
})

lambda_g_optimal <- lambdas[which.min(rmses_g)] # select optimal penalty


# #####################################################
# compute effects using optimal lambdas
# #####################################################

# compute movie effects
movie_avgs <- train_set %>%
  group_by(movieId) %>%
  summarize(b_m = sum(rating - mu)/(n()+lambda_m_optimal))

# compute user effects
user_avgs <- train_set %>%
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - mu - b_m)/(n() + lambda_u_optimal))

# compute genre effects
genre_avgs <- train_set %>% left_join(movie_avgs,by = "movieId") %>%
  left_join(user_avgs, by = 'userId') %>% 
  group_by(genres) %>% 
  summarize(b_g = sum(rating - mu - b_m - b_u)/(n() + lambda_g_optimal))

# predict ratings on test set
predicted_ratings <- test_set %>% 
  mutate(quarter = round_date(as_datetime(timestamp), "quarter"))%>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(genre_avgs, by = 'genres') %>%
  left_join(quarter_avgs, by = 'quarter') %>%
  mutate(pred = mu + b_m + b_u + b_g) %>%
  pull(pred)
predicted_ratings[is.na(predicted_ratings)] <- mu
rmse_test <- RMSE(predicted_ratings, test_set$rating)


# #####################################################
# compute RMSE for predictions on validation set
# #####################################################

predicted_ratings <- validation %>% 
  mutate(quarter = round_date(as_datetime(timestamp), "quarter")) %>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(genre_avgs, by = 'genres') %>%
  left_join(quarter_avgs, by = 'quarter') %>%
  mutate(pred = mu + b_m + b_u + b_g + b_q) %>% pull(pred)

predicted_ratings[is.na(predicted_ratings)] <- mu

rmse_validation <- RMSE(predicted_ratings, validation$rating)


# #####################################################
# save objects file for upload in Rmd
# #####################################################

save(edx, lambdas, lambda_m_optimal, lambda_u_optimal, lambda_g_optimal, rmse_test,
     rmse_validation, test_set, train_set, validation, file = "rmd_objects.RData")

# #####################################################
# plots used in Rmd
# #####################################################
# ave ratings by genres (movies with only 1 genre)
# edx %>% filter(genres %in% c("Action", "Adventure", "Animation", "Children", 
#   "Comedy", "Drama", "Fantasy", "IMAX", "Musical", "Sci-Fi", "Thriller", "War")) %>%  
#   group_by(genres) %>% summarize(avg_rating = mean(rating)) %>% 
# mutate(genres = reorder(genres, avg_rating)) %>% 
#   ggplot(aes(genres, avg_rating)) + geom_col() + 
# theme(axis.text.x = element_text(angle = 90, hjust = 1)) + 
#   labs(x = "Genre", y = "Average")
# 
# # average ratings from users giving >500 ratings
# edx %>% group_by(userId) %>% filter(n()>500) %>% 
# summarize(user = "Users with more than 500 ratings", avg_rating = mean(rating)) %>% 
#   ggplot(aes(user, avg_rating)) + geom_boxplot() + 
# geom_jitter(width = 0.03, alpha = .1) + xlab("") + ylab("Average rating")
# 
# # low ratings histogram
# edx %>% group_by(genres) %>% mutate(count = n()) %>% ungroup() %>% 
# filter(count < 25) %>% group_by(count) %>% summarize(nratings = n()) %>% 
# ggplot(aes(count, nratings, fill = count)) + geom_bar(stat = "identity") + 
# labs(x = "Number of Ratings", y = "Number of Genres") + 
#   scale_fill_gradient2(low = "red3", high = "forestgreen", mid = "grey79", 
# midpoint = 16, breaks=c(3,23),labels=c("Worse","Better")) + 
# theme(legend.title = element_blank()) 
#   
# train_set %>% mutate(week = round_date(as_datetime(timestamp), unit = "week")) %>%
#   group_by(week) %>%
#   summarize(rating = mean(rating)) %>%
#   ggplot(aes(week, rating)) +
#   geom_point() +
#   geom_smooth() + labs(x = "Time")

