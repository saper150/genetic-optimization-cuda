using System;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;


class Option {
    public string Name;
    public float[] Splits;
}

class Program {

    static string BuildPath(string inputFile, string type) {
        var fileName = Path.GetFileNameWithoutExtension(inputFile) + '-' + type + Path.GetExtension(inputFile);
        return Path.Combine("./data", Path.GetFileNameWithoutExtension(inputFile), fileName);
    }


    static void CreateDataSets(Option inputDataFile) {
        var reader = new Reader(inputDataFile.Name);
        // DataSetHelper.Rescale(reader.dataSet);

        DataSetHelper.Normalize(reader.dataSet);

        DataSetHelper.Shuffle(reader.dataSet);
        var splited = DataSetHelper.Split(reader.dataSet, inputDataFile.Splits);

        DataSetHelper.SaveInto(splited[0], BuildPath(inputDataFile.Name, "train"));
        DataSetHelper.SaveInto(splited[1], BuildPath(inputDataFile.Name, "test"));


        using (var writer = new StreamWriter(BuildPath(inputDataFile.Name, "meta"))) {
            writer.WriteLine("attributes count: " + reader.attributeCount);
            writer.WriteLine("labels count: " + reader.labelToId.Count());
        }

    }

    static void Main(string[] args) {

        Thread.CurrentThread.CurrentCulture = CultureInfo.CreateSpecificCulture("en-GB");
        Option[] files = new Option[] {
            new Option() { Name = "dataSets/iris.csv", Splits = new float[] { 0.8f, 0.2f } },
            new Option() { Name = "dataSets/magic.csv", Splits = new float[] { 0.05f, 0.015f } },
            new Option() { Name = "dataSets/banana.csv", Splits = new float[] { 0.05f, 0.015f } },
            new Option() { Name = "dataSets/ring.csv", Splits = new float[] { 00.05f, 0.015f } },
            new Option() { Name = "dataSets/spambase.csv", Splits = new float[] { 0.05f, 0.015f } }
        };
        Parallel.ForEach(files, file => {
            CreateDataSets(file);
        });

        // CreateDataSets("dataSets/magic.csv");
        // CreateDataSets("dataSets/banana.csv");
        // CreateDataSets("dataSets/ring.csv");
        // CreateDataSets("dataSets/spambase.csv");
    }
}
