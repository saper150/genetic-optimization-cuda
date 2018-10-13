using System;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;


class Program {

    static string BuildPath(string inputFile, string type) {
        var fileName = Path.GetFileNameWithoutExtension(inputFile) + '-' + type + Path.GetExtension(inputFile);
        return Path.Combine("./data", Path.GetFileNameWithoutExtension(inputFile), fileName);
    }

    static void CreateDataSets(string inputDataFile) {
        var reader = new Reader(inputDataFile);
        // DataSetHelper.Rescale(reader.dataSet);
        DataSetHelper.Normalize(reader.dataSet);

        // DataSetHelper.Shuffle(reader.dataSet);
        var splited = DataSetHelper.Split(reader.dataSet, new float[] { 0.5f, 0.3f, 0.2f });

        DataSetHelper.SaveInto(splited[0], BuildPath(inputDataFile, "train"));
        DataSetHelper.SaveInto(splited[1], BuildPath(inputDataFile, "test"));
        DataSetHelper.SaveInto(splited[2], BuildPath(inputDataFile, "verify"));

    }

    static void Main(string[] args) {

        Thread.CurrentThread.CurrentCulture = CultureInfo.CreateSpecificCulture("en-GB");
        CreateDataSets("iris.csv");
        // Console.WriteLine("Hello World!");
    }
}
