using Microsoft.ML;
using Microsoft.ML.AutoML;
using Microsoft.ML.Data;
using QuestpondML.Lab1.Model;
using System.Globalization;
using Microsoft.ML.Transforms.TimeSeries;

namespace QuestpondML.Lab1.Labs
{
    internal class NiftyEstimator
    {
        const string _dataPath = "Data\\Nifty 50 Historical Data.csv";

        /// <summary>
        /// Predicts Nifty 50 index using Fast Forest Regression
        /// </summary>
        public static void RunReplWithFastForrest()
        {
            var mlContext = new MLContext();
            string filePath = System.IO.Path.Combine(AppDomain.CurrentDomain.BaseDirectory, _dataPath);

            //Skip first 6 rows as they will have missing lag values
            var data = addLagPrices(LoadData(mlContext, filePath), 6).Skip(6).ToList();

            var splitData = mlContext.Data.TrainTestSplit(mlContext.Data.LoadFromEnumerable(data), testFraction: 0.05);
            var trainData = splitData.TrainSet;
            var testData = splitData.TestSet;                       

            long trainRowCount = mlContext.Data.CreateEnumerable<HistoricalStockPrice>(trainData, reuseRowObject: true).Count();
            long testRowCount = mlContext.Data.CreateEnumerable<HistoricalStockPrice>(testData, reuseRowObject: true).Count();

            PrintHeader("Fast Forest Regression", (int)trainRowCount, (int)testRowCount);            

            var pipeline = mlContext.Transforms
             .Concatenate("Features", new string[] { "LagPrice0", "LagPrice1", "LagPrice2", "LagPrice3", "LagPrice4", "LagPrice5" })
             .Append(mlContext.Regression.Trainers.FastForest("Price"));

            var trainedModel = pipeline.Fit(trainData);

            var engine = mlContext.Model.CreatePredictionEngine<HistoricalStockPrice, IndexData>(trainedModel);
            IndexData result = engine.Predict(new HistoricalStockPrice
            {
                Date = data[data.Count - 1].Date.AddMonths(1),
                LagPrice0 = data[data.Count - 1].Price,
                LagPrice1 = data[data.Count - 2].Price,
                LagPrice2 = data[data.Count - 3].Price,
                LagPrice3 = data[data.Count - 4].Price,
                LagPrice4 = data[data.Count - 5].Price,
                LagPrice5 = data[data.Count - 6].Price,
            });

            Console.WriteLine($"Predicted Price for next period: {data[data.Count - 1].Date.AddMonths(1)} {Math.Exp(result.Price)}");

            IDataView predictions = trainedModel.Transform(testData);
            var metrics = mlContext.Regression.Evaluate(predictions, "Price");

            PrintPredictions(mlContext, predictions, data.Skip((int)(trainRowCount)).ToList());          
            PrintRegressionMetrics(metrics);            
        }

        /// <summary>
        /// Runs AutoML experiment on Nifty 50 historical data
        /// </summary>
        public static void RunAutoMLExperiment()
        {
            var mlContext = new MLContext();
            string filePath = System.IO.Path.Combine(AppDomain.CurrentDomain.BaseDirectory, _dataPath);
            var data = addLagPrices(LoadData(mlContext, filePath), 6);

            PrintHeader("AutoML Experiment", data.Count, data.Count - 300);

            var result = RunAutoML(mlContext,
                mlContext.Data.LoadFromEnumerable(data),
                mlContext.Data.LoadFromEnumerable(data.Skip(300))
            );

            foreach (var run in result.RunDetails)
            {                
                Console.WriteLine($"Model: {run.TrainerName}");
                PrintRegressionMetrics(run.ValidationMetrics);             
            }

            // Get best model
            var bestModel = result.BestRun.Model;

            /*
            var engine = bestModel.CreateTimeSeriesEngine<HistoricalStockPrice, HistoricalStockPrice>(mlContext);
            var result2 = engine.Predict(new HistoricalStockPrice { Date = data.Last().Date.AddMonths(1) });
            Console.WriteLine(Math.Exp(data.Last().Price) + ":" + Math.Exp(result2.Price));
            */

            Console.WriteLine($"Best Model: {result.BestRun.TrainerName}");
        }

        private static ExperimentResult<RegressionMetrics>? RunAutoML(MLContext mlContext, IDataView trainData, IDataView testData)
        {
            var experimentSettings = new RegressionExperimentSettings
            {
                MaxExperimentTimeInSeconds = 120,
                OptimizingMetric = RegressionMetric.RSquared
            };
            var experiment = mlContext.Auto().CreateRegressionExperiment(experimentSettings);
            var result = experiment.Execute(trainData, testData, labelColumnName: "Price");
            return result;
        }


        /// <summary>
        /// Runs Support Vector Regression on Nifty 50 historical data
        /// </summary>
        public static void RunSVR()
        {
            var mlContext = new MLContext();
            string filePath = System.IO.Path.Combine(AppDomain.CurrentDomain.BaseDirectory, _dataPath);

            //Skip first 6 rows as they will have missing lag values
            var data = addLagPrices(LoadData(mlContext, filePath, useLogPricing: true), 6).Skip(6).ToList();                        

            int testPeriod = 6;                   
            var trainData = data.Take(data.Count() - testPeriod).ToList();
            var testData = data.Skip(data.Count()-testPeriod).ToList();

            long trainRowCount = data.Count() - testPeriod;
            long testRowCount = testPeriod;

            PrintHeader("SVR", (int)trainRowCount, (int)testRowCount);
            
            var dataProcessPipeline = mlContext.Transforms.Concatenate("Features", new string[] { "LagPrice0", "LagPrice1", "LagPrice2", "LagPrice3", "LagPrice4", "LagPrice5" })
                                        .Append(mlContext.Transforms.NormalizeMinMax("Features", "Features"));

            var trainer = mlContext.Regression.Trainers.LbfgsPoissonRegression(labelColumnName: "Price");

            var trainingPipeline = dataProcessPipeline.Append(trainer);

            ITransformer trainedModel = trainingPipeline.Fit(mlContext.Data.LoadFromEnumerable(trainData));

            // Make predictions on the test data
            IDataView predictions = trainedModel.Transform(mlContext.Data.LoadFromEnumerable(testData));                       
            var metrics = mlContext.Regression.Evaluate(predictions,"Price");            

            PrintPredictions(mlContext, predictions, testData);
            PrintRegressionMetrics(metrics);                        
        }


        /// <summary>
        /// Runs Singular Spectrum Analysis on Nifty 50 historical data    
        /// </summary>
        public static void RunSsa()
        {
            var mlContext = new MLContext();
            string filePath = System.IO.Path.Combine(AppDomain.CurrentDomain.BaseDirectory, _dataPath);
            var data = addLagPrices(LoadData(mlContext, filePath), 6);

            int testPeriod = 6;
            int trainSize = data.Count - testPeriod;

            PrintHeader("SSA", trainSize, testPeriod);

            var pipeline = mlContext.Forecasting.ForecastBySsa(
                outputColumnName: "ForecastedPrice",
                inputColumnName: "Price",  //Can experiment with LagPrice0, LagPrice1 etc. as well
                windowSize: 12,
                seriesLength: 120,
                trainSize: trainSize,
                horizon: testPeriod,
                confidenceLevel: 0.95f,
                confidenceLowerBoundColumn: "LowerBoundPrice",
                confidenceUpperBoundColumn: "UpperBoundPrice"
            );

            var trainData = mlContext.Data.LoadFromEnumerable(data.Take(trainSize));
            var testData = data.Skip(trainSize).ToList();

            var model = pipeline.Fit(trainData);

            var engine = model.CreateTimeSeriesEngine<HistoricalStockPrice, ForecastOutput>(mlContext);

            var forecast = engine.Predict();

            float[] predicted = forecast.ForecastedPrice.Select(x => Convert.ToSingle(Math.Exp(x))).ToArray(); // Convert back from log scale
            float[] actual = testData.Select(x => Convert.ToSingle(Math.Exp(x.Price))).ToArray();            // Convert back from log scale 
            
            // Compute metrics
            float mae = actual.Zip(predicted, (a, p) => Math.Abs(a - p)).Average();
            float rmse = (float)Math.Sqrt(actual.Zip(predicted, (a, p) =>
                Math.Pow(a - p, 2)).Average());
            float mape = actual.Zip(predicted, (a, p) =>
            {
                return Math.Abs((a - p) / a);
            }).Average() * 100f;

            for (int i = 0; i < testPeriod; i++)
            {
                Console.WriteLine($"Actual: {actual[i]}, Predicted: {predicted[i]}, LowerBound: {Math.Exp(forecast.LowerBoundPrice[i])}, UpperBound: {Math.Exp(forecast.UpperBoundPrice[i])}");
            }

            // Output
            Console.WriteLine("\n====================== METRICS ==================================\n");
            Console.WriteLine($"MAE:  {mae}");
            Console.WriteLine($"RMSE: {rmse}");
            Console.WriteLine($"MAPE: {mape}%");
            Console.WriteLine("=============================================================");
        }

        #region Console Display Functions
        static void PrintHeader(string title, int trainRows, int testRows)
        {
            Console.WriteLine("=============================================================");
            Console.WriteLine($"============= Starting {title} Training ====================");
            Console.WriteLine($"Train rows: {trainRows}, Test rows: {testRows}");   
            Console.WriteLine("=============================================================");
        }

        static void PrintPredictions(MLContext mlContext, IDataView predictions, List<HistoricalStockPrice> testData)
        {
            Console.WriteLine("====================== PREDICTIONS ==============================");
            int i = 0;
            foreach (var row in mlContext.Data.CreateEnumerable<IndexData>(predictions, reuseRowObject: false))
            {
                var o = testData[i++];
                Console.WriteLine($"Date: {o?.Date.ToString("MM-yyyy")}, Actual Price: {Math.Exp(o?.Price ?? 0)}, Predicted Price: {Math.Exp(row.Price)}");
            }
            Console.WriteLine("\n===============================================================\n");
        }

        static void PrintRegressionMetrics(RegressionMetrics metrics)
        {
            Console.WriteLine("====================== METRICS ==================================\n");
            Console.WriteLine($"R-Squared: {metrics.RSquared:0.##}");
            Console.WriteLine($"Mean Absolute Error: {metrics.MeanAbsoluteError:#.##}");
            Console.WriteLine($"Mean Squared Error: {metrics.MeanSquaredError:#.##}");
            Console.WriteLine($"Root Mean Squared Error: {metrics.RootMeanSquaredError:#.##}");
            Console.WriteLine("\n===============================================================\n");
        }
        #endregion

        #region Data Preparation    
        /// <summary>
        /// Prepares the data by loading from CSV and parsing date strings
        /// </summary>
        /// <param name="mlContext"></param>
        /// <param name="filePath"></param>
        /// <param name="useLogPricing">Take logarithm of the price instead of the actual price</param>
        /// <returns></returns>
        static List<HistoricalStockPrice> LoadData(MLContext mlContext, string filePath, bool useLogPricing=true)
        {
            System.Text.RegularExpressions.Regex regex = new System.Text.RegularExpressions.Regex(@"[^0-9.]");

            // Load the data into an IDataView, keeping the date as a string
            IDataView dataView = mlContext.Data.LoadFromTextFile<ModelInputString>(
                path: filePath,
                separatorChar: ',',
                hasHeader: true,
                allowQuoting: true
            );

            Action<ModelInputString, HistoricalStockPrice> transformation = (input, output) =>
            {
                // Parse the string date using the exact format and culture (Invariant Culture is good for consistency)
                output.Date = DateTime.ParseExact(
                      s: input.DateString ?? "",
                        format: "dd-MM-yyyy",
                    provider: CultureInfo.InvariantCulture
                );

                Single price = Convert.ToSingle(regex.Replace(input.Value ?? "", ""));
                output.Price = useLogPricing?Convert.ToSingle(Math.Log(price)):price;
            };

            // Apply the custom mapping
            var transformedDataView = mlContext.Transforms.CustomMapping<ModelInputString, HistoricalStockPrice>(
                transformation,
                "DateParser"
            ).Fit(dataView).Transform(dataView);


            return mlContext.Data.CreateEnumerable<HistoricalStockPrice>(transformedDataView, reuseRowObject: false)
                    .OrderBy(data => data.Date)
                    .ToList();
        }


        static List<HistoricalStockPrice> addLagPrices(List<HistoricalStockPrice> data, int numLags)
        {
            for (int i = 0; i < data.Count; i++)
            {
                for (int lag = 1; lag <= numLags; lag++)
                {
                    float laggedValue = (i - lag) >= 0 ? data[i - lag].Price : 0f;
                    switch (lag)
                    {
                        case 1:
                            data[i].LagPrice0 = laggedValue;
                            break;
                        case 2:
                            data[i].LagPrice1 = laggedValue;
                            break;
                        case 3:
                            data[i].LagPrice2 = laggedValue;
                            break;
                        case 4:
                            data[i].LagPrice3 = laggedValue;
                            break;
                        case 5:
                            data[i].LagPrice4 = laggedValue;
                            break;
                        case 6:
                            data[i].LagPrice5 = laggedValue;
                            break;
                    }
                }
            }
            return data;
        }
        #endregion

        #region Models
        public class IndexData
        {
            [ColumnName("Score")]
            public float Price { get; set; }
        }

        public class ForecastOutput
        {
            public float[] ForecastedPrice { get; set; }
            public float[] LowerBoundPrice { get; set; }
            public float[] UpperBoundPrice { get; set; }
        }
        #endregion
    }
}
