import weka.core.Attribute;
import weka.core.Instances;
import weka.attributeSelection.InfoGainAttributeEval;
import weka.attributeSelection.Ranker;
import weka.attributeSelection.AttributeSelection;

public class infoGain {
    public static String infoGainEval(Instances data) throws Exception {

        InfoGainAttributeEval evaluator = new InfoGainAttributeEval();
        Ranker ranker = new Ranker();

        AttributeSelection selector = new AttributeSelection();
        selector.setEvaluator(evaluator);
        selector.setSearch(ranker);
        selector.SelectAttributes(data);

        StringBuilder attrs = new StringBuilder();

        for (int i = 0; i < data.numAttributes() - 1; i++) {
            double score = evaluator.evaluateAttribute(i);
            if (score <= 0.01) {
                attrs.append(i).append(", ");
            }
        }

        if (attrs.length() > 0) {
            attrs.setLength(attrs.length() - 2);
        }

        return attrs.toString();
    }
}
