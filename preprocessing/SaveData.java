import weka.core.Instances;
import java.io.FileWriter;

public class SaveData {
    public static void save(Instances data, String filePath) throws Exception {
        FileWriter writer = new FileWriter(filePath);
        writer.write(data.toString());
        writer.close();
    }
}
