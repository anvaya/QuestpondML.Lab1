using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML.Data;

namespace QuestpondML.Lab1.Model
{
    internal class ModelInputString
    {
        [LoadColumn(0)]
        public string? DateString { get; set; }
        [LoadColumn(1)]
        public string? Value { get; set; }
    }

    internal class HistoricalStockPrice
    {                
        public DateTime Date { get; set; }

        public float fDate { get { return Date.Ticks; } }

        public float Price { get; set; }

        public float LagPrice0 { get; set; }
        public float LagPrice1 { get; set; }
        public float LagPrice2 { get; set; }

        public float LagPrice3 { get; set; }
        public float LagPrice4 { get; set; }
        public float LagPrice5 { get; set; }

        public float rsi { get; set; }   
    }


    internal class StockPricePrediction
    {        
        public float[] Price { get; set; } = [];
        public float[] PriceLowerBound { get; set; } = [];
        public float[] PriceUpperBound { get; set; } = [];
    }
}
