using System;


class Program
{
    static void Main(string[] args)
    {
        // Entrenar el modelo
        Console.WriteLine("Iniciando entrenamiento...");
        SentimentModelTrainer.Train();

        // Evaluar el modelo
        Console.WriteLine("Evaluando el modelo...");
        SentimentModelTrainer.Evaluate();

        // Probar una predicción
        string textoDePrueba = "I can't keep going like this,everything feels pointless.";
        var prediccion = SentimentModelTrainer.Predict(textoDePrueba);

        Console.WriteLine($"Texto: {textoDePrueba}");
        Console.WriteLine($"Predicción: {prediccion.Prediction}");
        Console.WriteLine($"Probabilidad: {prediccion.Probability}");
    }
}

