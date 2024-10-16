using Microsoft.ML;
using Microsoft.ML.Data;
using SentimentAnalyzerModel.Models;

public class SentimentModelTrainer
{
     private static readonly string DataPath = Path.Combine(Directory.GetCurrentDirectory(), "Data", "depression_dataset.csv");

    private static readonly string ModelPath = "MLModels/MLModel1.mbconfig";


    public static void Train()
    {
        var mlContext = new MLContext();

        var data = mlContext.Data.LoadFromTextFile<ModelInput>(DataPath, hasHeader: true, separatorChar: ',');

        var dataProcessPipeline = mlContext.Transforms.Text.FeaturizeText("Features", nameof(ModelInput.Text));

        var trainer = mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: "Label", featureColumnName: "Features");
        var trainingPipeline = dataProcessPipeline.Append(trainer);

        var trainedModel = trainingPipeline.Fit(data);

        mlContext.Model.Save(trainedModel, data.Schema, ModelPath);
    }
    public static void Evaluate()
    {
        var mlContext = new MLContext();

        // Cargar los datos desde el CSV para evaluación
        var data = mlContext.Data.LoadFromTextFile<ModelInput>(DataPath, hasHeader: true, separatorChar: ',');

        // Cargar el modelo entrenado
        var trainedModel = mlContext.Model.Load(ModelPath, out var modelInputSchema);

        // Transformar los datos para evaluar
        var predictions = trainedModel.Transform(data);

        // Evaluar el modelo
        var metrics = mlContext.BinaryClassification.Evaluate(predictions, labelColumnName: "Label");

        // Mostrar los resultados de la evaluación
        Console.WriteLine($"Exactitud: {metrics.Accuracy:P2}");
        Console.WriteLine($"AUC: {metrics.AreaUnderRocCurve:P2}");
        Console.WriteLine($"F1 Score: {metrics.F1Score:P2}");
    }

    public static ModelOutput Predict(string text)
    {
        var mlContext = new MLContext();

     
        var trainedModel = mlContext.Model.Load(ModelPath, out var modelInputSchema);

     
        var predictionEngine = mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(trainedModel);

    
        var input = new ModelInput { Text = text };

        
        var prediction = predictionEngine.Predict(input);
        return prediction;
    }

}
