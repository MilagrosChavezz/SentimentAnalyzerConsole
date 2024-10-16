﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SentimentAnalyzerModel.Models
{
  
        public class ModelOutput
        {
            public bool Prediction { get; set; }
            public float Probability { get; set; }
            public float Score { get; set; }
        }

    
}
