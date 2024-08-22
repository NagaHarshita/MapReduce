import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.StringTokenizer;
import java.util.Arrays;


public class BigramIndex {

    public static class BigramMapper extends Mapper<Object, Text, Text, Text> {

        private Text bigram = new Text();
        private Text docId = new Text();


        @Override
        protected void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            String[] textSplit = value.toString().split("\\t", 2);
            docId.set(textSplit[0]);

            String[] bigrams = {"computer science","information retrieval","power politics","los angeles","bruce willis"};

            String preprocessedString = textSplit[1].replaceAll("[^a-zA-Z]+", " ").toLowerCase();
            StringTokenizer tokenizer = new StringTokenizer(preprocessedString, " ");
            String first = null;
            String second = null;
            while (tokenizer.hasMoreTokens()) {
                String curWord = tokenizer.nextToken();
                if (!curWord.trim().isEmpty()) {
                    if (first == null) {
                        first = curWord;
                        continue;
                    }

                    if (second == null) {
                        second = curWord;
                    } else {
                        first = second;
                        second = curWord;
                    }

                    if (Arrays.asList(bigrams).contains(first + " " + second)){
                        bigram.set(String.format("%s %s", first, second));
                        context.write(bigram, docId);
                    }
                    
                }
            }
        }
    }

    public static class BigramReducer extends Reducer<Text, Text, Text, Text> {

        @Override
        public void reduce(Text word, Iterable<Text> docIds, Context context) throws IOException, InterruptedException {
            Map<String, Integer> freqMap = new HashMap<>();
            for (Text docId : docIds) {
                String docIdString = docId.toString();
                freqMap.put(docIdString, freqMap.getOrDefault(docIdString, 0) + 1);
            }
  
            StringBuilder docIdWordFreq = new StringBuilder();
            for (Map.Entry<String, Integer> entry : freqMap.entrySet()) {
                if (docIdWordFreq.length() > 0) {
                    docIdWordFreq.append(" ");
                }
                String wordfreqStr = String.format("%s:%d", entry.getKey(), entry.getValue());
                docIdWordFreq.append(wordfreqStr);
            }
  
            context.write(word, new Text(docIdWordFreq.toString()));
        }
    }

    public static void main(String[] args) throws IOException, ClassNotFoundException, InterruptedException {
        Configuration conf = new Configuration();
        Job job = Job.getInstance(conf, "Bigram: Inverted Index");
        job.setJarByClass(BigramIndex.class);
        job.setMapperClass(BigramMapper.class);
        job.setReducerClass(BigramReducer.class);

        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class);

        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));

        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}