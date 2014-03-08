package com.twitter.algebra.nmf;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Scanner;
import java.util.Set;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.lib.input.MultipleInputs;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.math.CardinalityException;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.SequentialAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.hadoop.DistributedRowMatrix;
import org.jboss.netty.handler.codec.http.HttpHeaders.Values;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.myabandeh.algebra.AlgebraCommon;
import com.myabandeh.algebra.AlgebraCommon.ExcludeMetaFilesFilter;
import com.myabandeh.algebra.matrix.format.MapDir;
import com.myabandeh.algebra.matrix.format.MatrixOutputFormat;
import com.myabandeh.algebra.matrix.format.RowPartitioner;
import com.myabandeh.algebra.matrix.multiply.AtBOuterStaticMapsideJoinJob;
import com.twitter.algebra.nmf.NMFCommon;

/**
 * Perform A x B matrix multiplication
 * 
 * Approach: Outer-join
 * 
 * Number of jobs: 1
 * 
 * Assumption: (1) Transpose At is already available, (2) the smaller of At and
 * B is in {@link MapDir} format, (3) the bigger of At and B is in partially
 * sorted sequence file format--Partial sort meaning that the input to each
 * mapper is sorted.
 * 
 * Design: Iterate over the rows of the bigger of At and B. For each row, load
 * the corresponding row from the other matrix, which is efficient since the
 * other is in {@link MapDir} format and the input rows are partially sorted.
 * Each mapper perform Ati x Bi (row i of At and row i of B) multiplication and
 * generates partial matrix Ci. The reducers sum up partial Ci matrices to get C
 * = A x B.
 */
public class DMJ extends AbstractJob {
  private static final Logger log = LoggerFactory.getLogger(DMJ.class);

  public static final String MATRIXINMEMORY = "matrixInMemory";
  /**
   * The smaller of the matrices is chosen to be loaded at runtime via its
   * {@link MapDir} format. This config parameter specifies which will be loaded
   * as {@link MapDir}.
   */
  public static final String AISMAPDIR = "matrix.a.is.mapdir";
  public static final String RESULTROWS = "matrix.result.num.rows";
  public static final String RESULTCOLS = "matrix.result.num.cols";
  public static final String PARTITIONCOLS = "matrix.partition.num.cols";

  @Override
  public int run(String[] strings) throws Exception {
    addOutputOption();
    addOption("numColsAt", "nca",
        "Number of columns of the first input matrix", true);
    addOption("numColsB", "ncb",
        "Number of columns of the second input matrix", true);
    addOption("atMatrix", "atMatrix", "The first matrix, transposed");
    addOption("bMatrix", "bMatrix", "The second matrix");
    Map<String, List<String>> parsedArgs = parseArguments(strings);
    if (parsedArgs == null) {
      return -1;
    }

    Path aPath = new Path(getOption("atMatrix"));
    Path bPath = new Path(getOption("bMatrix"));
    int atCols = Integer.parseInt(getOption("numColsAt"));
    int bCols = Integer.parseInt(getOption("numColsB"));
    run(getConf(), aPath, bPath, getOutputPath(), atCols, bCols, bCols, true, 1);
    return 0;
  }

  /**
   * Perform A x B, where At and B are already wrapped in a DistributedRowMatrix
   * object. Refer to {@link DMJ} for further details.
   * 
   * @param conf the initial configuration
   * @param At transpose of matrix A
   * @param B matrix B
   * @param label the label for the output directory
   * @param numberOfJobs the hint for the desired number of parallel jobs
   * @return AxB wrapped in a DistributedRowMatrix object
   * @throws IOException
   * @throws InterruptedException
   * @throws ClassNotFoundException
   */
  public static DistributedRowMatrix run(Configuration conf,
      DistributedRowMatrix At, DistributedRowMatrix B, int partitionCols, String label,
      boolean useCombiner, int numberOfJobs) throws IOException,
      InterruptedException, ClassNotFoundException {
    log.info("running " + DMJ.class.getName());
    if (At.numRows() != B.numRows()) {
      throw new CardinalityException(At.numRows(), B.numRows());
    }
    Path outPath = new Path(At.getOutputTempPath(), label);
    FileSystem fs = FileSystem.get(outPath.toUri(), conf);
    DMJ job = new DMJ();
    if (!fs.exists(outPath)) {
      job.run(conf, At.getRowPath(), B.getRowPath(), outPath, At.numCols(),
          B.numCols(), partitionCols, useCombiner, numberOfJobs);
    } else {
      log.warn("----------- Skip already exists: " + outPath);
    }
    DistributedRowMatrix distRes =
        new DistributedRowMatrix(outPath, At.getOutputTempPath(), At.numCols(),
            B.numCols());
    distRes.setConf(conf);
    return distRes;
  }

  /**
   * Perform A x B, where A and B refer to the paths that contain matrices in
   * {@link SequenceFileInputFormat}. The smaller of At and B must also conform
   * with {@link MapDir} format. Refer to {@link DMJ} for further details.
   * 
   * @param conf the initial configuration
   * @param atPath path to transpose of matrix A.
   * @param bPath path to matrix B
   * @param matrixOutputPath path to which AxB will be written
   * @param atCols number of columns of At (rows of A)
   * @param numberOfJobs the hint for the desired number of parallel jobs
   * @throws IOException
   * @throws InterruptedException
   * @throws ClassNotFoundException
   */
  public void run(Configuration conf, Path atPath, Path bPath,
      Path matrixOutputPath, int atCols, int bCols, int partitionCols, boolean useCombiner,
      int numberOfJobs) throws IOException, InterruptedException,
      ClassNotFoundException {
    FileSystem fs = FileSystem.get(atPath.toUri(), conf);
    long atSize = MapDir.du(atPath, fs);
    long bSize = MapDir.du(bPath, fs);
    log.info("Choosing the smaller matrix: atSize: " + atSize + " bSize: "
        + bSize);
    boolean aIsMapDir = atSize < bSize;
    if (aIsMapDir)
      runJobsInParallel(conf, atPath, bPath, matrixOutputPath, atCols, bCols, partitionCols,
          aIsMapDir, useCombiner, numberOfJobs);
    else
      runJobsInParallel(conf, bPath, atPath, matrixOutputPath, atCols, bCols, partitionCols,
          aIsMapDir, useCombiner, numberOfJobs);
  }

  /**
   * Split a big job into multiple smaller jobs. Each job should be more
   * efficient as it puts less load on the reducers. We also can run the jobs in
   * parallel.
   * 
   * The maximum number of jobs is the number of column partitions produced by
   * the {@link ColPartitionJob}.
   * 
   * @param conf
   * @param mapDirPath the path to the matrix in MapDir format
   * @param matrixInputPath the input matrix path that we iterate on
   * @param matrixOutputPath the output matrix path
   * @param atCols number of columns in At
   * @param aIsMapDir is A chosen to be loaded as MapDir
   * @param numberOfJobs the hint for the desired number of parallel jobs
   * @throws IOException
   * @throws InterruptedException
   * @throws ClassNotFoundException
   */
  private static void runJobsInParallel(Configuration conf, Path mapDirPath,
      Path matrixInputPath, Path matrixOutputPath, int atCols, int bCols, int partitionCols, 
      boolean aIsMapDir, boolean useCombiner, int numberOfJobs)
      throws IOException, InterruptedException, ClassNotFoundException {
    if (numberOfJobs == 1) {
      DMJ job = new DMJ();
      Job hjob =
          job.run(conf, mapDirPath, new Path[] { matrixInputPath },
              matrixOutputPath, atCols, bCols, partitionCols, aIsMapDir, useCombiner);
      boolean res = hjob.waitForCompletion(true);
      if (!res)
        throw new IOException("Job failed! ");
      return;
    }

    FileSystem fs = FileSystem.get(matrixOutputPath.toUri(), conf);
    FileStatus[] files =
        fs.listStatus(mapDirPath, new AlgebraCommon.ExcludeMetaFilesFilter());
    int numColPartitions = files.length;

    // use the hint and max feasible parallelism to find an optimum degree for
    // parallelism
    numberOfJobs = numColPartitions;

    // run the jobs
    Job[] jobs = new Job[numberOfJobs];
    int nextPartitionIndex = 0;
    for (int jobIndex = 0; jobIndex < numberOfJobs; jobIndex++) {
      DMJ job = new DMJ();
      Path subJobOutPath = new Path(matrixOutputPath, "" + jobIndex);

      jobs[jobIndex] =
          job.run(conf, new Path(mapDirPath, "" + jobIndex),
              new Path[] { matrixInputPath }, subJobOutPath, atCols, bCols, partitionCols, 
              aIsMapDir, useCombiner);
      // boolean res = jobs[jobIndex].waitForCompletion(true);
      // if (!res)
      // throw new IOException("Job failed! " + jobIndex);
    }

    // wait for the jobs (in case they are run in parallel and move their output
    // to the main output directory
    for (int jobIndex = 0; jobIndex < numberOfJobs; jobIndex++) {
      boolean res = jobs[jobIndex].waitForCompletion(true);
      if (!res)
        throw new IOException("Job failed! " + jobIndex);
      // Path subJobDir = new Path(matrixOutputPath, "" + jobIndex);
      // FileStatus[] jobOutFiles = fs.listStatus(subJobDir,
      // new ExcludeMetaFilesFilter());
      // for (FileStatus jobOutFile : jobOutFiles) {
      // Path src = jobOutFile.getPath();
      // Path dst = new Path(matrixOutputPath, src.getName());
      // // unique name by indexing with folder id
      // log.info("fs.rename " + src + " -> " + dst);
      // fs.rename(src, dst);
      // }
      // log.info("fs.delete " + subJobDir);
      // fs.delete(subJobDir, true);
    }
  }

  /**
   * How many column partitions are generated by the transpose job? The
   * partition number is embedded in the file name
   * 
   * @param files the files produced by the transpose job
   * @return number of column partitions in the files
   */
  private static int computeNumOfColPartitions(FileStatus[] files) {
    Set<Integer> partitionSet = new HashSet<Integer>();
    for (FileStatus fileStatus : files) {
      String fileName = fileStatus.getPath().getName();// part-cp-1-r-00002-j-10
      Scanner scanner = new Scanner(fileName);
      scanner.useDelimiter("-");
      scanner.next();// part
      scanner.next();// cp
      String filePartitionStr = scanner.next();
      int filePartitionIndex = Integer.parseInt(filePartitionStr);
      partitionSet.add(filePartitionIndex);
    }
    return partitionSet.size();
  }

  /**
   * Filter the files belong to a column partition to the list. The partition
   * number is embedded in the file name.
   * 
   * @param files the input files
   * @param partitionIndex the index of the column partition
   * @param inFilesList the result list of filtered files
   */
  @SuppressWarnings("deprecation")
  private static void addFilesOfAPartition(FileStatus[] files,
      int partitionIndex, List<Path> inFilesList) {
    for (FileStatus fileStatus : files) {
      String fileName = fileStatus.getPath().getName();// part-cp-1-r-00002-j-10
      Scanner scanner = new Scanner(fileName);
      scanner.useDelimiter("-");
      scanner.next();// part
      scanner.next();// cp
      String filePartitionStr = scanner.next();
      int filePartitionIndex = Integer.parseInt(filePartitionStr);
      if (filePartitionIndex == partitionIndex) {
        if (fileStatus.isDir())// mapfile
          inFilesList.add(new Path(fileStatus.getPath(), "data"));
        else
          inFilesList.add(fileStatus.getPath());
      }
    }
  }

  /**
   * Perform A x B, where At and B refer to the paths that contain matrices in
   * {@link SequenceFileInputFormat}. One of At and B must also conform with
   * {@link MapDir} format. Refer to {@link DMJ} for further details.
   * 
   * @param conf the initial configuration
   * @param mapDirPath path to the matrix in {@link MapDir} format
   * @param matrixInputPaths the list of paths to matrix input partitions over
   *          which we iterate
   * @param matrixOutputPath path to which AxB will be written
   * @param atCols number of columns of At (rows of A)
   * @param aIsMapDir is A chosen to be loaded as MapDir
   * @param numberOfJobs the hint for the desired number of parallel jobs
   * @return the running job
   * @throws IOException
   * @throws InterruptedException
   * @throws ClassNotFoundException
   */
  public Job run(Configuration conf, Path mapDirPath, Path[] matrixInputPaths,
      Path matrixOutputPath, int atCols, int bCols, int partitionCols, boolean aIsMapDir,
      boolean useCombiner) throws IOException, InterruptedException,
      ClassNotFoundException {
    conf.set(MATRIXINMEMORY, mapDirPath.toString());
    conf.setBoolean(AISMAPDIR, aIsMapDir);
    conf.setInt(RESULTROWS, atCols);
    conf.setInt(RESULTCOLS, bCols);
    conf.setInt(PARTITIONCOLS, partitionCols);
    FileSystem fs = FileSystem.get(matrixOutputPath.toUri(), conf);
    NMFCommon.setNumberOfMapSlots(conf, fs, matrixInputPaths, "dmj");

    @SuppressWarnings("deprecation")
    Job job = new Job(conf);
    job.setJarByClass(DMJ.class);
    job.setJobName(DMJ.class.getSimpleName());
    matrixOutputPath = fs.makeQualified(matrixOutputPath);

    for (Path path : matrixInputPaths) {
      path = fs.makeQualified(path);
      MultipleInputs.addInputPath(job, path, SequenceFileInputFormat.class);
    }

    FileOutputFormat.setOutputPath(job, matrixOutputPath);
    job.setMapperClass(MyMapper.class);
    job.setMapOutputKeyClass(IntWritable.class);
    job.setMapOutputValueClass(VectorWritable.class);

    if (useCombiner)
      job.setCombinerClass(AtBOuterStaticMapsideJoinJob.MyReducer.class);

    int numReducers = NMFCommon.getNumberOfReduceSlots(conf, "dmj");
    job.setNumReduceTasks(numReducers);// TODO: make it a parameter
    // ensures total order (when used with {@link MatrixOutputFormat}),
    RowPartitioner.setPartitioner(job, RowPartitioner.IntRowPartitioner.class,
        atCols);

    job.setReducerClass(AtBOuterStaticMapsideJoinJob.MyReducer.class);
    job.setOutputFormatClass(MatrixOutputFormat.class);
    job.setOutputKeyClass(IntWritable.class);
    job.setOutputValueClass(VectorWritable.class);
    job.submit();
    return job;
  }

  /**
   * Iterate over the input vectors, do an outer join with the corresponding
   * vector from the other matrix, and write the resulting partial matrix to the
   * reducers.
   * 
   * @author myabandeh
   * 
   */
  public static class MyMapper extends
      Mapper<IntWritable, VectorWritable, IntWritable, VectorWritable> {
    private MapDir otherMapDir;
    private boolean aIsMapDir;
    private boolean useInMemCombiner = true;
    private VectorWritable otherVectorw = new VectorWritable();
    private VectorWritable outVectorw = new VectorWritable();
    private IntWritable outIntw = new IntWritable();
    private double[] outValues = null;
    int resultRows, resultCols;
    int partitionCols;

    @Override
    public void setup(Context context) throws IOException {
      Configuration conf = context.getConfiguration();
      Path inMemMatrixPath = new Path(conf.get(MATRIXINMEMORY));
      resultRows = conf.getInt(RESULTROWS, Integer.MAX_VALUE);
      resultCols = conf.getInt(RESULTCOLS, Integer.MAX_VALUE);
      partitionCols = conf.getInt(PARTITIONCOLS, Integer.MAX_VALUE);
      otherMapDir = new MapDir(conf, inMemMatrixPath);
      aIsMapDir = conf.getBoolean(AISMAPDIR, true);
      if (useInMemCombiner) {
        try {
          outValues = new double[resultRows * partitionCols];
        } catch (Exception e) {
          outValues = null;
          System.gc();
          useInMemCombiner = false;
          System.err.println("Not enough mem for in memory combiner of size "
              + resultRows * partitionCols);
        }
      }
    }

    @Override
    public void map(IntWritable index, VectorWritable bvw, Context context)
        throws IOException, InterruptedException {
      Vector normalInput = bvw.get();
      Writable res = otherMapDir.get(index, otherVectorw);
      if (res == null) {
        // too many nulls could indicate a bug, good to check
        context.getCounter("MapDir", "nullValues").increment(1);
        return;
      }
      Vector mapDirVector = otherVectorw.get();
      if (aIsMapDir)
        outerProduct(mapDirVector, normalInput, context);
      else
        outerProduct(normalInput, mapDirVector, context);
    }

    void outerProduct(Vector aVector, Vector bVector, Context context)
        throws IOException, InterruptedException {
      Iterator<Vector.Element> it = aVector.nonZeroes().iterator();
      while (it.hasNext()) {
        Vector.Element e = it.next();
        if (useInMemCombiner)
          combineInMem(e.index(), e.get(), bVector);
        else {
          outIntw.set(e.index());
          outVectorw.set(bVector.times(e.get()));
          context.write(outIntw, outVectorw);
        }
      }
    }

    private void combineInMem(int row, double scale, Vector bVector) {
      Iterator<Vector.Element> it = bVector.nonZeroes().iterator();
      while (it.hasNext()) {
        Vector.Element e = it.next();
        int localCol = globalToLocalCol(e.index());
        int localIndex = rowcolToIndex(row, localCol);
        outValues[localIndex] += scale * e.get();
      }
    }

    private int minGlobalCol = Integer.MIN_VALUE;
    private int globalToLocalCol(int col) {
      int localCol = col % partitionCols;
      if (minGlobalCol == Integer.MIN_VALUE)
        minGlobalCol = col - localCol;
      return localCol;
    }

    private int rowcolToIndex(int row, int localCol) {
      return row * partitionCols + localCol;
    }

    private void dumpInMemMatrix(Context context) throws IOException {
      if (minGlobalCol == Integer.MIN_VALUE)
        throw new IOException("minGlobalCol is not set!");
      IntWritable iw = new IntWritable();
      VectorWritable vw = new VectorWritable();
      RandomAccessSparseVector vector =
          new RandomAccessSparseVector(resultCols, partitionCols);
      for (int r = 0; r < resultRows; r++) {
        int index = rowcolToIndex(r, 0);
        boolean hasNonZero = false;
        for (int c = 0; c < partitionCols; c++, index++) {
          double value = outValues[index];
          if (value != 0) {
            hasNonZero = true;
            vector.set(c + minGlobalCol, value);
          }
        }
        if (!hasNonZero)
          continue;
        iw.set(r);
        vw.set(new SequentialAccessSparseVector(vector));
        try {
          context.write(iw, vw);
        } catch (InterruptedException e) {
          context.getCounter("Error", "interrup").increment(1);
        }
        vector = new RandomAccessSparseVector(resultCols, partitionCols);
      }
    }

    @Override
    public void cleanup(Context context) throws IOException {
      otherMapDir.close();
      if (useInMemCombiner)
        dumpInMemMatrix(context);
    }

  }

}
