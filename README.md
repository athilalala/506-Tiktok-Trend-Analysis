# 506 – TikTok Trend Analysis  
## TikTok Virality Prediction Using Multimodal Data


## Project Description
This project investigates the factors influencing TikTok video virality using a multimodal data science approach. The study integrates Natural Language Processing (NLP) to analyze captions, titles, and hashtags, Computer Vision techniques to examine video content, and early engagement signals such as initial views, likes, and comments.  

The objective is to identify key drivers of virality and evaluate whether combining textual, visual, and engagement-based features improves the accuracy of predicting video performance.


## Project Goals

- Successfully predict whether a TikTok video will perform above average using caption text, hashtags, visual content, and early engagement metrics within the first **24–72 hours** after posting.  
- Evaluate the contribution of computer vision–derived visual features alongside textual and engagement data to improve virality prediction.



## Data Collection Plan

### Potential Data Sources
- Public TikTok datasets from platforms such as Kaggle and HuggingFace containing captions, hashtags, engagement metrics, and video metadata.

### Data Collection Method
- Primary data will be obtained from publicly available datasets.  
- If required, additional publicly accessible TikTok content will be collected using ethical scraping tools to obtain captions, engagement metrics, and video links.



## Proposed Timeline

- Data Collection (1 week): Identification and acquisition of multimodal TikTok datasets (text, engagement metrics, video content).  
- Data Preprocessing (1 week): Text cleaning, handling missing values, normalization of engagement metrics, and video frame preparation.  
- Feature Engineering (3 weeks): Extraction of NLP features, computer vision features from video frames, and early engagement indicators.  
- Exploratory Analysis & Visualization (1 week): Descriptive analysis, trend visualization, and feature relationship assessment.  
- Model Development & Evaluation (2 weeks): Multimodal model training, performance evaluation, and comparative analysis.



## 如何切换 VSCode 内 GitHub Copilot 登录账号

如果需要在 VSCode 中切换 GitHub Copilot 所使用的 GitHub 账号，请按照以下步骤操作：

### 方法一：通过账号菜单切换

1. 打开 VSCode，点击左下角的 **账号图标**（人形图标）。
2. 在弹出的菜单中找到当前已登录的 GitHub 账号。
3. 点击该账号旁边的 **"注销"（Sign Out）** 选项，退出当前账号。
4. 再次点击账号图标，选择 **"使用 GitHub 登录以使用 GitHub Copilot"**，使用新账号重新登录。

### 方法二：通过命令面板切换

1. 按下 `Ctrl+Shift+P`（macOS 上为 `Cmd+Shift+P`）打开 **命令面板**。
2. 输入并选择 `GitHub Copilot: Sign Out`，退出当前账号。
3. 再次打开命令面板，输入并选择 `GitHub Copilot: Sign In`，使用新账号登录。

### 注意事项

- 切换账号后，VSCode 将使用新登录账号的 Copilot 订阅。
- 若新账号没有有效的 Copilot 订阅，将无法使用 Copilot 功能。
- 如遇登录问题，可尝试重启 VSCode 后再进行上述操作。






