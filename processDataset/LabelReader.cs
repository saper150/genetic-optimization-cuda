using System.Collections.Generic;
using System.IO;
using System.Linq;
using CsvHelper;

class Reader {

    public CudaDataSet<int> dataSet;

    public Dictionary<string, int> labelToId = new Dictionary<string, int>();

    public Reader(string filePath) {

        var variables = new List<float[]>();
        var labels = new List<string>();
        var csv = new CsvReader(new StreamReader(filePath));
        csv.Configuration.Delimiter = ",";
        csv.Configuration.DetectColumnCountChanges = true;
        csv.Read();
        int variablesCount = csv.Context.ColumnCount;
        while (csv.Read()) {
            variables.Add(new float[csv.Context.ColumnCount - 1]);
            for (int i = 0; i < csv.Context.ColumnCount - 1; i++) {
                variables.Last()[i] = csv.GetField<float>(i);
            }
            labels.Add(csv.GetField<string>(variablesCount - 1));
        }
        labelToId = labels.Distinct()
            .Select((val, index) => new { val, index })
            .ToDictionary(x => x.val, x => x.index);

        dataSet = new CudaDataSet<int>() {
            Vectors = new FlattArray<float>(variables.ToArray()),
            Classes = labels.Select(x => labelToId[x]).ToArray(),
        };
        dataSet.orginalIndeces = DataSetHelper.CreateIndeces(dataSet);

    }

}
