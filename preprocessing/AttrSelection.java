
//load required classes
import weka.attributeSelection.CfsSubsetEval;
import weka.attributeSelection.GreedyStepwise;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;

public class AttrSelection {
    public static Instances select(Instances data) throws Exception {
        // create AttributeSelection objects
        AttributeSelection filter = new AttributeSelection();

        // Create evaluation and search algorithm objects
        CfsSubsetEval eval = new CfsSubsetEval();
        GreedyStepwise search = new GreedyStepwise();

        // set the algorithm to search backward
        search.setSearchBackwards(true);

        // set the filter to use the evaluator and search algorithm
        filter.setEvaluator(eval);
        filter.setSearch(search);

        // specify the dataset
        filter.setInputFormat(data);
        // apply
        return Filter.useFilter(data, filter);
    }
}