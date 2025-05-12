import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Normalize;

public class NormalizeData {
    public static Instances Normalize(Instances data) throws Exception {
        Normalize filter = new Normalize();
        filter.setInputFormat(data);
        return Filter.useFilter(data, filter);
    }
}
