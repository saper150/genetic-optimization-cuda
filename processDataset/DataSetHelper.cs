
using System;
using System.IO;
using System.Linq;
using System.Threading.Tasks;

public class CudaDataSet<T> {
    public FlattArray<float> Vectors;
    public T[] Classes;
    public int[] orginalIndeces;
}


public static class DataSetHelper {
    static Random rand = new Random();

    public static void Shuffle<T>(CudaDataSet<T> data) {

        for (int i = 0; i < data.Classes.Length; i++) {
            var toSwap = rand.Next(data.Classes.Length - i);
            data.Swap(toSwap, i);
        }

    }


    public static CudaDataSet<T>[] Split<T>(CudaDataSet<T> data, float[] parts) {
        var result = new CudaDataSet<T>[parts.Length];
        var splitedClasses = Split(data.Classes, parts);
        var splitedVectors = Split(data.Vectors, parts);
        var splitedIndeces = Split(data.orginalIndeces, parts);

        for (int i = 0; i < splitedClasses.Length; i++) {
            result[i] = new CudaDataSet<T>() {
                Vectors = splitedVectors[i],
                Classes = splitedClasses[i],
                orginalIndeces = splitedIndeces[i]
            };
        }
        return result;
    }

    public static void SaveInto(CudaDataSet<int> data, string path) {
        FileInfo fileInfo = new FileInfo(path);

        if (!fileInfo.Exists) {
            Directory.CreateDirectory(fileInfo.Directory.FullName);
        }
        var d = data.Vectors.To2d();
        using (var writer = new StreamWriter(path)) {
            for (int i = 0; i < d.Length; i++) {
                var toSave = String.Join(',', d[i].Select(x => x.ToString())) + ',' + data.Classes[i].ToString();
                writer.WriteLine(toSave);
            }
        }
    }


    public static FlattArray<T>[] Split<T>(FlattArray<T> data, float[] parts) {
        var splited = new FlattArray<T>[parts.Length];
        int startIndex = 0;

        for (int i = 0; i < parts.Length; i++) {
            int endIndex = startIndex +
                (int)Math.Round(data.GetLength(0) * parts[i]);

            if (endIndex >= data.GetLength(0)) {
                endIndex = data.GetLength(0);
            }
            int len = endIndex - startIndex;

            T[] arr = new T[len * data.GetLength(1)];
            Array.Copy(data.Raw,
                startIndex * data.GetLength(1),
                arr,
                 0,
                 len * data.GetLength(1)
                );
            startIndex = endIndex;
            splited[i] = new FlattArray<T>(arr, data.GetLength(1));


        }
        return splited;
    }


    public static T[][] Split<T>(T[] data, float[] parts) {
        var splited = new T[parts.Length][];
        int startIndex = 0;

        for (int i = 0; i < parts.Length; i++) {
            int endIndex = startIndex +
                (int)Math.Round(data.Length * parts[i]);
            if (endIndex >= data.Length) {
                endIndex = data.Length;
            }
            int len = endIndex - startIndex;
            splited[i] = new T[len];
            Array.Copy(data, startIndex, splited[i], 0, len);
            startIndex = endIndex;

        }
        return splited;
    }


    private static void Swap<T>(this CudaDataSet<T> data, int row1, int row2) {
        data.Vectors.Swap(row1, row2);
        data.Classes.Swap(row1, row2);
        data.orginalIndeces.Swap(row1, row2);
    }

    public static void StandardScore<T>(CudaDataSet<T> data) {

        var vectors = data.Vectors;
        float[] avg = ColumnsAvrages(data);

        float[] standardDeviation = new float[vectors.GetLength(0)];

        for (int row = 0; row < standardDeviation.GetLength(0); row++) {
            for (int col = 0; col < vectors.GetLength(1); col++) {
                var difference = vectors[row, col] - avg[col];
                standardDeviation[col] += difference * difference;
            }
        }

        for (int i = 0; i < vectors.GetLength(1); i++) {
            standardDeviation[i] = (float)Math.Sqrt(standardDeviation[i] / vectors.GetLength(0));
        }

        for (int row = 0; row < standardDeviation.GetLength(0); row++) {
            for (int col = 0; col < vectors.GetLength(1); col++) {
                vectors[row, col] = (vectors[row, col] - avg[col]) / standardDeviation[col];
            }
        }
    }

    public static void Rescale<T>(CudaDataSet<T> data) {
        for (int col = 0; col < data.Vectors.GetLength(1); col++) {
            var (min, max) = ColumnExtrema(data, col);
            for (int row = 0; row < data.Vectors.GetLength(0); row++) {
                data.Vectors[row, col] = (data.Vectors[row, col] - min) / (max - min);
            }
        }
    }

    public static (float min, float max) ColumnExtrema<T>(CudaDataSet<T> data, int column) {
        float max = float.MinValue;
        float min = float.MaxValue;
        for (int i = 0; i < data.Vectors.GetLength(0); i++) {
            if (data.Vectors.Get(i, column) > max) {
                max = data.Vectors.Get(i, column);
            }
            if (data.Vectors.Get(i, column) < min) {
                min = data.Vectors.Get(i, column);
            }
        }
        return (min, max);
    }

    public static float[] Variances(FlattArray<float> vectors) {

        float[] variances = new float[vectors.GetLength(1)];
        Parallel.For(0, vectors.GetLength(1), col => {
            float avrage = ColumnAvrage(vectors, col);
            float[] diffrencesSquared = new float[vectors.GetLength(0)];
            for (int row = 0; row < vectors.GetLength(0); row++) {
                float f = vectors[row, col] - avrage;
                diffrencesSquared[row] = f * f;
            }
            variances[col] = diffrencesSquared.Average();

        });

        return variances;
    }


    public static float ColumnAvrage(FlattArray<float> data, int column) {
        float avg = 0;
        for (int i = 0; i < data.GetLength(0); i++) {
            avg += data[i, column];
        }
        return avg / data.GetLength(0);

    }

    public static float[] ColumnsAvrages<T>(CudaDataSet<T> data) {

        var vectors = data.Vectors;
        float[] avg = new float[vectors.GetLength(1)];

        for (int i = 0; i < vectors.GetLength(1); i++) {
            avg[i] = ColumnAvrage(vectors, i);
        }

        return avg;
    }

    public static int[] CreateIndeces<T>(CudaDataSet<T> set) {
        int len = set.Classes.Length;
        int[] res = new int[len];

        for (int i = 0; i < len; i++) {
            res[i] = i;
        }

        return res;
    }

}