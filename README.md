# Livestream-Ecommerce-Recommender-System (LERS)

doi: [10.6814/NCCU202201098](http://thesis.lib.nccu.edu.tw/cgi-bin/gs32/gsweb.cgi?o=dallcdr&s=id=%22G0109356002%22.&searchmode=basic)

## Abstract

In recent years, live stream e-commerce shopping has received extensive attention from e-commerce businesses and streaming platforms. Different from traditional TV shopping and online shopping, the emerging products roll out continuously on the live stream shopping platform where users and streamers interact and synchronize in real-time. Such a dynamic environment forms a complex user context. The recommender system plays a crucial role in assisting users in information-seeking tasks and decision-making from information overload. Previous recommender systems mainly focus on optimizing accuracy, which results in filter bubbles problem and high churn rates in the long run. To balance exploration and exploitation (EE) trade-off under a dynamic and fast-changing recommendation context, the research formulates the problem as a contextual bandit problem. This study provides a reinforcement learning (RL)-based solution for a new business scenario (i.e., live stream e-commerce) which addresses three relationships between customers, streamers, and products in both static and temporal user contexts. We use Gated Recurrent Unit (GRU) to model the context changes in users' preferences in streamers and products while maintaining their long-term engagement. By encoded uncertainty in neural networks with Variational Autoencoder (VAE) for user modeling and Bayesian Neural Network (BNN) for a product recommendation, the proposed Live E-commerce Recommender System (LERS) can control the balance of EE trade-off. To the best of our knowledge, our study is the first neural network-based contextual bandit algorithm dealing with the recommendation problem in the live streaming e-commerce platforms. We compared our algorithm with classic multi-armed bandit algorithms including UCB1, LinUCB, Exp3, and NeuralUCB. Preliminary experiment results on real-world data corroborate our theory and shed light on potential applications of our algorithm to real-world business problems.

## Contents

Acknowledgements  i </br>
摘要  ii </br> 
Abstract  iii </br>
Contents  v </br>
List of Figures  viii </br>
List of Tables  x </br>
1 Introduction  1 </br>
2 RelatedWork  4 </br>
2.1 Live Streaming E-commerce  4 </br>
2.2 Recommender Systems  5 </br>
2.3 Live Streaming Recommender System  6 </br>
2.4 Contextual Multi-armed Bandit Methods  8 </br>
2.5 Uncertainty Modeling  10 </br>
3 The Proposed Framework  12 </br>
3.1 Problem Definition  12 </br>
3.2 Framework Overview  13 </br>
3.3 Gated Recurrent Unit Networks in Temporal Context Model  16 </br>
3.4 Variational Autoencoder for Blurry Context  18 </br>
3.5 Bayesian Neural Networks for Exploring Product Recommendation  20 </br>
3.6 Training Procedure  21 </br>
4 Experiments  25 </br>
4.1 Datasets  25 </br>
4.2 Implementation Environment  25 </br>
4.3 Customer Context Features  26 </br>
4.3.1 Static Context Features  26  </br>
4.3.2 Customer-Product Context Features  26  </br>
4.3.3 Customer-Streamer Context Features  27 </br>
4.4 Temporal Context Modeling  28 </br>
4.4.1 RNN-based Models for Temporal Context  28 </br>
4.4.2 Identify the Appropriate Sequence Length of Temporal Context  29 </br>
4.5 Full Context Analysis  33 </br>
4.6 Dimension Reduction Analysis  35 </br>
4.7 Production Recommendation Analysis  36 </br>
4.7.1 Evaluation Metrics  36 </br>
4.7.2 Experiment Dataset  38 </br>
4.7.3 Recommendation Context for Product Recommendation  40 </br>
4.7.4 Temporal Context for Product Recommendation  42 </br>
4.7.5 End-to-End Live E-commerce Recommender System  44 </br>
4.8 Algorithm Comparison Experiments  45 </br>
4.8.1 Experiments Settings  46 </br>
4.8.2 Normal Dataset  47 </br>
4.8.3 Active Dataset  50 </br>
4.8.4 Repeat Dataset  51 </br>
5 Discussion  54 </br>
5.1 Offline Environment  54 </br>
5.2 Feature Enrichment  54 </br>
5.3 Context Engineering  55 </br>
5.4 Neural Network  55 </br>
6 Conclusion  57 </br>
References  59 </br>
