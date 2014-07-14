package com.twitter.algebra.matrix.multiply;

import java.io.IOException;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.join.CompositeInputFormat;
import org.apache.hadoop.mapreduce.lib.join.TupleWritable;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.math.CardinalityException;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.SequentialAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.function.Functions;
import org.apache.mahout.math.hadoop.DistributedRowMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.twitter.algebra.matrix.format.MatrixOutputFormat;

/**
 * Perform A x B matrix multiplication
 * 
 * Approach: Outer-join (borrowed from Mahout's {@link MatrixMultiplicationJob})
 * 
 * Number of jobs: 1
 * 
 * Assumption: (1) Transpose At is already available, (2) At and B have the
 * exactly the same partitioning (number of partitions and entries inside each
 * partition), (3) the entries inside each partition are sorted, (4) At and B
 * have different number of columns (to distinguish them in map side join).
 * 
 * Design: Hadoop put the partitions with the same index together and iterates
 * over the entries of both partition at the same time (mapside join). Each mapper
 * perform Ati x Bi (row i of At and row i of B) multiplication and generates
 * partial matrix Ci. The reducers sum up partial Ci matrices to get C = A x B.
 */
public class AtBOuterStaticMapsideJoinJob extends AbstractJob {
  private static final Logger log = LoggerFactory
      .getLogger(AtBOuterStaticMapsideJoinJob.class);

  static final String OUT_CARD = "output.vector.cardinality";

  @Override
  public int run(String[] strings) throws Exception {
    addOutputOption();
    addOption("numColsB", "ncb",
        "Number of columns of the second input matrix", true);
    addOption("inputPathA", "ia", "Path to the first input matrix", true);
    addOption("inputPathB", "ib", "Path to the second input matrix", true);

    Map<String, List<String>> argMap = parseArguments(strings);
    if (argMap == null) {
      return -1;
    }
    run(getConf(), new Path(getOption("inputPathA")), new Path(
        getOption("inputPathB")), getOutputPath(),
        Integer.parseInt(getOption("numColsB")));
    return 0;
  }

  public void run(Configuration conf, Path atPath, Path bPath, Path outPath,
      int outCardinality) throws IOException, InterruptedException,
      ClassNotFoundException {
    conf.setInt(OUT_CARD, outCardinality);
    @SuppressWarnings("deprecation")
    Job job = new Job(conf);
    job.setJobName(AtBOuterStaticMapsideJoinJob.class.getSimpleName());
    job.setJarByClass(AtBOuterStaticMapsideJoinJob.class);

    FileSystem fs = FileSystem.get(atPath.toUri(), conf);
    atPath = fs.makeQualified(atPath);
    bPath = fs.makeQualified(bPath);
    job.setInputFormatClass(CompositeInputFormat.class);
    //mapside join expression
    job.getConfiguration().set(
        CompositeInputFormat.JOIN_EXPR,
        CompositeInputFormat.compose("inner", SequenceFileInputFormat.class,
            atPath, bPath));

    job.setOutputFormatClass(MatrixOutputFormat.class);
    outPath = fs.makeQualified(outPath);
    FileOutputFormat.setOutputPath(job, outPath);
    job.setMapperClass(MyMapper.class);
    job.setMapOutputKeyClass(IntWritable.class);
    job.setMapOutputValueClass(VectorWritable.class);
    
    job.setCombinerClass(MyReducer.class);
    
    int numReducers = conf.getInt("algebra.reduceslots.multiply", 10);
    job.setNumReduceTasks(numReducers);

    job.setReducerClass(MyReducer.class);
    job.setOutputKeyClass(IntWritable.class);
    job.setOutputValueClass(VectorWritable.class);
    job.submit();
    boolean res = job.waitForCompletion(true);
    if (!res)
      throw new IOException("Job failed");
  }

  public static DistributedRowMatrix run(Configuration conf, DistributedRowMatrix A,
      DistributedRowMatrix B, String label) throws IOException, InterruptedException, ClassNotFoundException {
    log.info("running " + AtBOuterStaticMapsideJoinJob.class.getName());
    if (A.numRows() != B.numRows()) {
      throw new CardinalityException(A.numRows(), B.numRows());
    }
    Path outPath = new Path(A.getOutputTempPath(), label);
    FileSystem fs = FileSystem.get(outPath.toUri(), conf);
    AtBOuterStaticMapsideJoinJob job = new AtBOuterStaticMapsideJoinJob();
    if (!fs.exists(outPath)) {
      job.run(conf, A.getRowPath(), B.getRowPath(), outPath, B.numCols());
    } else {
      log.warn("----------- Skip already exists: " + outPath);
    }
    DistributedRowMatrix distRes = new DistributedRowMatrix(outPath,
        A.getOutputTempPath(), A.numCols(), B.numCols());
    distRes.setConf(conf);
    return distRes;
  }
  
  public static class MyMapper extends
      Mapper<IntWritable, TupleWritable, IntWritable, VectorWritable> {
    private int outCardinality;
    private final IntWritable row = new IntWritable();

    @Override
    public void setup(Context context) throws IOException {
      outCardinality = context.getConfiguration().getInt(OUT_CARD,
          Integer.MAX_VALUE);
    }

    @Override
    public void map(IntWritable index, TupleWritable v, Context context)
        throws IOException, InterruptedException {
      boolean firstIsOutFrag = ((VectorWritable) v.get(0)).get().size() == outCardinality;
      Vector outFrag = firstIsOutFrag ? ((VectorWritable) v.get(0)).get()
          : ((VectorWritable) v.get(1)).get();
      Vector multiplier = firstIsOutFrag ? ((VectorWritable) v.get(1)).get()
          : ((VectorWritable) v.get(0)).get();

      VectorWritable outVector = new VectorWritable();
      Iterator<Vector.Element> it = multiplier.nonZeroes().iterator();
      while (it.hasNext()) {
        Vector.Element e = it.next();
        row.set(e.index());
        outVector.set(outFrag.times(e.get()));
        context.write(row, outVector);
      }
    }
  }

  public static class MyReducer extends
      Reducer<IntWritable,VectorWritable,IntWritable,VectorWritable> {
    private VectorWritable outvw = new VectorWritable();
    @Override
    public void reduce(IntWritable rowNum, Iterable<VectorWritable> values,
        Context context) throws IOException, InterruptedException {
      Iterator<VectorWritable> it = values.iterator();
      if (!it.hasNext())
        return;
      Vector accumulator = new RandomAccessSparseVector(it.next().get());
      while (it.hasNext()) {
        Vector row = it.next().get();
        accumulator.assign(row, Functions.PLUS);
      }
      outvw.set(new SequentialAccessSparseVector(accumulator));
      context.write(rowNum, outvw);
    }
  }
}
