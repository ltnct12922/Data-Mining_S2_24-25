import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.RemoveDuplicates;

public class removeDuplicates {
    public static Instances removeDup(Instances data) throws Exception {
        RemoveDuplicates filter = new RemoveDuplicates();
        filter.setInputFormat(data);
        Instances filteredData = Filter.useFilter(data, filter);
        return filteredData;
    }
}
