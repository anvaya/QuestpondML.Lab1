# QuestpondML.Lab1 🚀

[![.NET](https://img.shields.io/badge/.NET-8.0-purple.svg)](https://dotnet.microsoft.com/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE.txt)
[![ML.NET](https://img.shields.io/badge/ML.NET-5.0.0-green.svg)](https://dotnet.microsoft.com/apps/machinelearning-ai/ml-dotnet)

A comprehensive Machine Learning laboratory project implementing various time series forecasting algorithms for financial data prediction, specifically targeting the Nifty 50 stock index.

## 📋 Table of Contents

- [About](#-about)
- [Features](#-features)
- [Algorithms Implemented](#-algorithms-implemented)
- [Project Structure](#-project-structure)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Usage](#-usage)
- [Data](#-data)
- [Model Performance](#-model-performance)
- [Contributing](#-contributing)
- [License](#-license)
- [Author](#-author)

## 🎯 About

**QuestpondML.Lab1** is an educational and experimental platform developed by **Mrugendra (Yogesh) Bhure** as part of the Questpond AI/ML Cohort. This project demonstrates practical implementation of multiple machine learning algorithms for financial time series forecasting, with a focus on real-world stock market prediction using the Nifty 50 index.

The project serves as a comprehensive learning resource for:

- Financial time series analysis and forecasting
- Multiple ML algorithm comparison and evaluation
- Feature engineering for time series data
- Model evaluation using comprehensive metrics
- AutoML experimentation and automated model selection

## ✨ Features

### 🤖 Machine Learning Algorithms
- **Fast Forest Regression** - Ensemble method for non-linear pattern detection
- **Singular Spectrum Analysis (SSA)** - Advanced time series decomposition with confidence intervals
- **Support Vector Regression (SVR)** - Kernel-based regression approach
- **AutoML Experimentation** - Automated model discovery and optimization

### 📊 Data Processing
- **Lag Feature Engineering** - Creates temporal features (6 lag periods)
- **Relative Strength Index (RSI)** - Technical indicator for momentum analysis
- **Data Preprocessing** - Custom parsing and normalization
- **Train/Test Splitting** - Proper temporal validation
- **Logarithmic Transformation** - Optional scaling for exponential patterns

### 📈 Evaluation Metrics
- R² (Coefficient of Determination)
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Percentage Error (MAPE)

## 🧠 Algorithms Implemented

### 1. Fast Forest Regression
- **Type**: Ensemble method using decision trees
- **Best for**: Non-linear relationships and complex patterns
- **Features**: 6 lag features as input, 95/5 train-test split

### 2. Singular Spectrum Analysis (SSA) ⭐ *Currently Active*
- **Type**: Time series decomposition method
- **Best for**: Advanced forecasting with confidence intervals
- **Configuration**: Window size: 12, Series length: 120
- **Output**: Point forecasts with upper/lower confidence bounds

### 3. Support Vector Regression (SVR)
- **Type**: Kernel-based regression
- **Best for**: High-dimensional feature spaces
- **Features**: 6 lag features + RSI, feature normalization, 6-period holdout validation

### 4. AutoML Experimentation
- **Type**: Automated model discovery
- **Optimization**: R² metric maximization
- **Runtime**: Limited to 120 seconds for efficiency

## 📁 Project Structure

```
QuestpondML.Lab1/
├── 📄 README.md                        # This file
├── 📄 LICENSE.txt                      # MIT License
├── 📄 QuestpondML.Lab1.sln             # Solution file
│
└── 📁 QuestpondML.Lab1/                # Main project
    ├── 📄 Program.cs                   # Main entry point
    ├── 📄 QuestpondML.Lab1.csproj      # Project configuration
    │
    ├── 📁 Labs/                        # ML implementations
    │   └── 📄 NiftyEstimator.cs        # Core ML algorithms
    │
    ├── 📁 Model/                       # Data models
    │   └── 📄 HistoricalStockPrice.cs  # Data structures
    │
    ├── 📁 Data/                        # Dataset
    │   └── 📄 Nifty 50 Historical Data.csv
    │
    ├── 📁 bin/                         # Build output
    ├── 📁 obj/                         # Build intermediates
    └── 📁 .vs/                         # VS configuration
```

## 🔧 Prerequisites

- **.NET 8.0 SDK** or higher
- **Visual Studio 2022** or compatible IDE
- **Windows OS** (project uses Windows-specific paths)
- **Git** for cloning

## 📦 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/anvaya/QuestpondML.Lab1.git
   cd QuestpondML.Lab1s
   ```

2. **Restore NuGet packages**
   ```bash
   dotnet restore
   ```

3. **Build the project**
   ```bash
   dotnet build
   ```

### Package Requirements

| Package | Version | Purpose |
|---------|---------|---------|
| `Microsoft.ML` | 5.0.0 | Core ML.NET framework |
| `Microsoft.ML.AutoML` | 0.23.0 | Automated ML capabilities |
| `Microsoft.ML.Mkl.Components` | 5.0.0 | Intel Math Kernel Library optimizations |

## 🚀 Usage

### Running the Application

1. **Navigate to the project directory**
   ```bash
   cd QuestpondML.Lab1
   ```

2. **Run the current experiment**
   ```bash
   dotnet run
   ```

3. **Switch between algorithms** (edit `Program.cs`):
   ```csharp
   // Currently active: SSA
   estimator.RunSsa();

   // Uncomment to use other algorithms:
   // estimator.RunAutoMLExperiment();
   // estimator.RunSVR();
   // estimator.RunReplWithFastForrest();
   ```

### Running Specific Experiments

#### Fast Forest Regression
```csharp
var estimator = new NiftyEstimator();
estimator.RunReplWithFastForrest();
```

#### Singular Spectrum Analysis
```csharp
var estimator = new NiftyEstimator();
estimator.RunSsa();
```

#### Support Vector Regression
```csharp
var estimator = new NiftyEstimator();
estimator.RunSVR();
```

#### AutoML Experiment
```csharp
var estimator = new NiftyEstimator();
estimator.RunAutoMLExperiment();
```

## 📊 Data

### Dataset Description
- **Source**: Nifty 50 Historical Data
- **Format**: CSV (Comma-separated values)
- **Columns**: `Date`, `Price`
- **Date Format**: `dd-MM-yyyy` (e.g., `01-05-2024`)
- **Time Range**: May 2024 to November 2025 (monthly data)
- **Price Format**: Comma thousands separator (e.g., `22,901.35`)

### Data Schema
```csv
Date,Price
01-05-2024,22,901.35
01-06-2024,23,110.45
...
```

### Enhanced Feature Set
The `HistoricalStockPrice` model includes the following features:
- **Date**: DateTime object for temporal ordering
- **Price**: Current price (log-transformed for modeling)
- **LagPrice0-5**: 6 lagged price features for temporal patterns
- **RSI**: Relative Strength Index (14-period default, configurable)

### Preprocessing Pipeline
1. **Date Parsing**: Converts `dd-MM-yyyy` format to `DateTime`
2. **Price Normalization**: Removes comma separators, converts to `float`
3. **Lag Feature Creation**: Generates 6 lagged price features
4. **RSI Calculation**: Computes Relative Strength Index with configurable period (default: 6)
5. **Optional Log Transformation**: Applies logarithmic scaling for model training

### 🔧 Technical Implementation: RSI Calculator

The `CalculateRSI` function implements the standard Relative Strength Index algorithm:

**Algorithm Details**:
- **Default Period**: 6 periods (configurable, optimized for monthly data)
- **Formula**: `RSI = 100 - (100 / (1 + RS))` where `RS = Average Gain / Average Loss`
- **Smoothing**: Uses smoothed moving average for RSI calculation
- **Log Scale Handling**: Automatically converts from logarithmic price data
- **Edge Cases**: Handles insufficient data gracefully (RSI = 0 for initial periods)

**Features**:
- Configurable calculation period
- Support for log-transformed price data
- Efficient memory usage with in-place calculations
- Robust handling of edge cases (zero losses, insufficient data)

## 📈 Model Performance

### Evaluation Metrics Output Example
```
=== Model Performance Metrics ===
R² Score: 0.8567
MAE: 245.32
MSE: 85,432.11
RMSE: 292.27
MAPE: 1.24%
```

### SSA Forecast Example
```
=== SSA Forecast Results ===
Period 1 Forecast: 23,450.67
Confidence Interval: [23,100.12, 23,801.22]
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Guidelines
- Follow C# coding conventions
- Add XML documentation for new methods
- Update this README for significant changes
- Ensure all NuGet packages are updated

## 📝 License

This project is licensed under the MIT License - see the [LICENSE.txt](LICENSE.txt) file for details.

## 👨‍💻 Author

**Mrugendra (Yogesh) Bhure**
- **Role**: Questpond AI/ML Cohort Member
- **Focus**: Machine Learning & Financial Time Series Analysis
- **Project**: QuestpondML Laboratory Experiments

---

**🎓 Educational Purpose**: This project is part of the Questpond AI/ML Cohort training program and serves as a practical demonstration of machine learning concepts applied to financial forecasting.

**⚠️ Disclaimer**: This project is for educational purposes only. The predictions and models should not be used for actual trading or investment decisions.