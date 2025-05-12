import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

public class removeAttr {
    public static Instances remove(Instances data, String indica) throws Exception {
        Remove filter = new Remove();
        filter.setAttributeIndices(indica);
        filter.setInputFormat(data);
        return Filter.useFilter(data, filter);
    }
}
