import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;

public class HandlemissValues {
    public static Instances handel(Instances data) throws Exception {
        ReplaceMissingValues filter = new ReplaceMissingValues();
        filter.setInputFormat(data); // set format for data like telling ML my data looks like this
        return Filter.useFilter(data, filter);
    }
}
