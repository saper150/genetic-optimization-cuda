using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using System.Xml.Linq;

class Program
{
    static string[] fields = new string[] { "id", "product_code", "product_name", "standard_cost", "list_price", "reorder_level", "target_level", "quantity_per_unit", "discontinued", "minimum_reorder_quantity", "category" };
    static Regex regex = new Regex("(\"id\":\"(\\d+)\",\"product_code\":\"(.*)\",\"product_name\":\"(.*)\",\"standard_cost\":\"(.*)\",\"list_price\":\"(.*)\",\"reorder_level\":\"(.*)\",\"target_level\":\"(.*)\",\"quantity_per_unit\":\"(.*)\",\"discontinued\":\"(.*)\",\"minimum_reorder_quantity\":\"(.*)\",\"category\":\"(.*)\")|((.{10})(.{64})(.{255})(.{9})(.{9})(.{4})(.{4})(.{128})(.{1})(.{3})(.{128}))|((.*),(.*),(.*),(.*),(.*),(.*),(.*),(.*),(.*),(.*),(.*),)");

    static Dictionary<string, string> line(string line)
    {
        var match = regex.Match(line);
        return new Dictionary<string, string>(
            fields.Select((field, i) => new KeyValuePair<string, string>
                (field, match.Groups[i + new int[] { 2, 14, 26 }.First(x => match.Groups[x].Success)].Value.Trim()))
        );
    }
    static void Main(string[] args)
    {
        File.WriteAllText("res.xml",
            new XElement("root",
                File.ReadAllLines("data.txt")
                    .Select(line)
                    .Select(x => new XElement("element", x.Select(y => new XElement(y.Key, y.Value))))
                )
                .ToString()
        );
    }
}
