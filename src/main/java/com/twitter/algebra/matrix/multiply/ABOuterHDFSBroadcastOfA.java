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
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.math.CardinalityException;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.hadoop.DistributedRowMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.twitter.algebra.AlgebraCommon;
import com.twitter.algebra.matrix.format.MatrixOutputFormat;
import com.twitter.algebra.matrix.format.RowPartitioner;

/**
 * Perform A x B matrix multiplication
 * 
 * Approach: Outer-join
 * 
 * Number of jobs: 1
 * 
 * Assumption: A is small enough to fit into the memory
 * 
 * Design: load A from HDFS into the mappers' memory. Each mapper perform A-i x
 * Bi- (col i of A and row i of B) multiplication and generates partial matrix
 * Ci. The reducers sum up partial Ci matrices to get C = A x B.
 */
public class ABOuterHDFSBroadcastOfA extends AbstractJob {
  private static final Logger log = LoggerFactory
      .getLogger(ABOuterHDFSBroadcastOfA.class);

  public static final String MATRIXINMEMORY = "matrixInMemory";
  public static final String MATRIXINMEMORYROWS = "memRows";
  public static final String MATRIXINMEMORYCOLS = "memCols";

  @Override
  public int run(String[] strings) throws Exception {
    addInputOption();
    addOutputOption();
    addOption(MATRIXINMEMORY, "times",
        "The name of the file that contains the matrix that fits into memory");
    addOption(MATRIXINMEMORYROWS, "r", "Number of rows of the other matrix");
    addOption(MATRIXINMEMORYCOLS, "c", "Number of cols of the other matrix");
    Map<String, List<String>> parsedArgs = parseArguments(strings);
    if (parsedArgs == null) {
      return -1;
    }

    String inMemMatrixFileName = getOption(MATRIXINMEMORY);
    int inMemMatrixNumRows = Integer.parseInt(getOption(MATRIXINMEMORYROWS));
    int inMemMatrixNumCols = Integer.parseInt(getOption(MATRIXINMEMORYCOLS));
    run(getConf(), inMemMatrixFileName, getInputPath(), getOutputPath(),
        inMemMatrixNumRows, inMemMatrixNumCols);
    return 0;
  }

  /**
   * Perform A x B, where A and B are already wrapped in a DistributedRowMatrix
   * object. Refer to {@link ABOuterHDFSBroadcastOfA} for further details.
   * 
   * @param conf
   *          the initial configuration
   * @param A
   *          matrix A
   * @param B
   *          matrix B
   * @param label
   *          the label for the output directory
   * @return AxB wrapped in a DistributedRowMatrix object
   * @throws IOException
   * @throws InterruptedException
   * @throws ClassNotFoundException
   */
  public static DistributedRowMatrix run(Configuration conf, DistributedRowMatrix A,
      DistributedRowMatrix B, String label) throws IOException, InterruptedException, ClassNotFoundException {
    log.info("running " + ABOuterHDFSBroadcastOfA.class.getName());
    if (A.numCols() != B.numRows()) {
      throw new CardinalityException(A.numCols(), B.numRows());
    }
    Path outPath = new Path(A.getOutputTempPath(), label);
    FileSystem fs = FileSystem.get(outPath.toUri(), conf);
    ABOuterHDFSBroadcastOfA job = new ABOuterHDFSBroadcastOfA();
    if (!fs.exists(outPath)) {
      job.run(conf, A.getRowPath(), B.getRowPath(), outPath, A.numRows(),
          A.numCols());
    } else {
      log.warn("----------- Skip already exists: " + outPath);
    }
    DistributedRowMatrix distRes = new DistributedRowMatrix(outPath,
        A.getOutputTempPath(), A.numRows(), B.numCols());
    distRes.setConf(conf);
    return distRes;
  }

  /**
   * Perform A x B, where A and B refer to the paths that contain matrices in
   * {@link SequenceFileInputFormat} Refer to {@link ABOuterHDFSBroadcastOfA}
   * for further details.
   * 
   * @param conf
   *          the initial configuration
   * @param matrixInputPath
   *          path to matrix A (must be small enough to fit into memory)
   * @param inMemMatrixDir
   *          path to matrix B 
   * @param matrixOutputPath
   *          path to which AxB will be written
   * @param inMemMatrixNumRows
   *          A rows
   * @param inMemMatrixNumCols
   *          A cols
   * @throws IOException
   * @throws InterruptedException
   * @throws ClassNotFoundException
   */
  public void run(Configuration conf, Path inMemMatrixDir,
      Path matrixInputPath, Path matrixOutputPath, int inMemMatrixNumRows,
      int inMemMatrixNumCols) throws IOException, InterruptedException,
      ClassNotFoundException {
    run(conf, inMemMatrixDir.toString(), matrixInputPath, matrixOutputPath,
        inMemMatrixNumRows, inMemMatrixNumCols);
  }

  /**
   * Perform A x B, where A and B refer to the paths that contain matrices in
   * {@link SequenceFileInputFormat} Refer to {@link ABOuterHDFSBroadcastOfA}
   * for further details.
   * 
   * @param conf
   *          the initial configuration
   * @param matrixInputPath
   *          path to matrix A
   * @param inMemMatrixDir
   *          path to matrix B (must be small enough to fit into memory)
   * @param matrixOutputPath
   *          path to which AxB will be written
   * @param inMemMatrixNumRows
   *          B rows
   * @param inMemMatrixNumCols
   *          B cols
   * @throws IOException
   * @throws InterruptedException
   * @throws ClassNotFoundException
   */
  public void run(Configuration conf, String inMemMatrixDir,
      Path matrixInputPath, Path matrixOutputPath, int inMemMatrixNumRows,
      int inMemMatrixNumCols) throws IOException, InterruptedException,
      ClassNotFoundException {
    conf.set(MATRIXINMEMORY, inMemMatrixDir);
    conf.setInt(MATRIXINMEMORYROWS, inMemMatrixNumRows);
    conf.setInt(MATRIXINMEMORYCOLS, inMemMatrixNumCols);
    @SuppressWarnings("deprecation")
    Job job = new Job(conf);
    job.setJarByClass(ABOuterHDFSBroadcastOfA.class);
    job.setJobName(ABOuterHDFSBroadcastOfA.class.getSimpleName());
    FileSystem fs = FileSystem.get(matrixInputPath.toUri(), conf);
    matrixInputPath = fs.makeQualified(matrixInputPath);
    matrixOutputPath = fs.makeQualified(matrixOutputPath);

    FileInputFormat.addInputPath(job, matrixInputPath);
    job.setInputFormatClass(SequenceFileInputFormat.class);
    FileOutputFormat.setOutputPath(job, matrixOutputPath);
    job.setMapperClass(MyMapper.class);
    job.setMapOutputKeyClass(IntWritable.class);
    job.setMapOutputValueClass(VectorWritable.class);

    // ensures total order (when used with {@link MatrixOutputFormat}),
    RowPartitioner.setPartitioner(job, RowPartitioner.IntRowPartitioner.class,
        inMemMatrixNumRows);

    job.setCombinerClass(AtBOuterStaticMapsideJoinJob.MyReducer.class);
    
    job.setReducerClass(AtBOuterStaticMapsideJoinJob.MyReducer.class);
    job.setOutputFormatClass(MatrixOutputFormat.class);
    job.setOutputKeyClass(IntWritable.class);
    job.setOutputValueClass(VectorWritable.class);
    
    job.submit();
    boolean res = job.waitForCompletion(true);
    if (!res)
      throw new IOException("Job failed!");
  }

  public static class MyMapper extends
      Mapper<IntWritable, VectorWritable, IntWritable, VectorWritable> {
    private DenseMatrix inMemMatrix;
    private VectorWritable outvw = new VectorWritable();
    private IntWritable outiw = new IntWritable();

    @Override
    public void setup(Context context) throws IOException {
      Configuration conf = context.getConfiguration();
      Path inMemMatrixPath = new Path(conf.get(MATRIXINMEMORY));
      int inMemMatrixNumRows = conf.getInt(MATRIXINMEMORYROWS, 0);
      int inMemMatrixNumCols = conf.getInt(MATRIXINMEMORYCOLS, 0);
      //TODO: Add support for in-memory sparse matrix
      inMemMatrix = AlgebraCommon.mapDirToDenseMatrix(inMemMatrixPath,
          inMemMatrixNumRows, inMemMatrixNumCols, conf);
    }

    @Override
    public void map(IntWritable index, VectorWritable vw, Context context)
        throws IOException, InterruptedException {
      Vector outFrag = vw.get();
      Vector multiplier = inMemMatrix.viewColumn(index.get());
      Iterator<Vector.Element> it = multiplier.nonZeroes().iterator();
      while (it.hasNext()) {
        Vector.Element e = it.next();
        outiw.set(e.index());
        outvw.set(outFrag.times(e.get()));
        context.write(outiw, outvw);
      }
    }
  }
}
