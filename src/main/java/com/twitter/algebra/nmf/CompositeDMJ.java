package com.twitter.algebra.nmf;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashSet;
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
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.hadoop.DistributedRowMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.myabandeh.algebra.AlgebraCommon;
import com.myabandeh.algebra.matrix.format.MapDir;
import com.myabandeh.algebra.matrix.format.MatrixOutputFormat;
import com.myabandeh.algebra.matrix.format.RowPartitioner;

/**
 * (A ./ (A*MEM+A.*a2+a1)) .* B Approach: Broadcast of MEM and Dynamic Mapside
 * join of B
 */
public class CompositeDMJ extends AbstractJob {
  private static final Logger log = LoggerFactory.getLogger(CompositeDMJ.class);

  public static final String MAPDIRMATRIX = "mapDirMatrix";
  
  public static final String MATRIXINMEMORY = "matrixInMemory";
  public static final String MATRIXINMEMORYROWS = "memRows";
  public static final String MATRIXINMEMORYCOLS = "memCols";

  public static final String ALPHA1 = "CompositeDMJ.alpha1";
  public static final String ALPHA2 = "CompositeDMJ.alpha2";

  /**
   * The smaller of the matrices is chosen to be loaded at runtime via its
   * {@link MapDir} format. This config parameter specifies which will be loaded
   * as {@link MapDir}.
   */
  public static final String AISMAPDIR = "matrix.a.is.mapdir";

  @Override
  public int run(String[] strings) throws Exception {
    addOutputOption();
    addOption("numColsAt", "nca",
        "Number of columns of the first input matrix", true);
    addOption("atMatrix", "atMatrix", "The first matrix, transposed");
    addOption("bMatrix", "bMatrix", "The second matrix");
    addOption(MATRIXINMEMORY, "times",
        "The name of the file that contains the matrix that fits into memory");
    addOption(MATRIXINMEMORYROWS, "r", "Number of rows of the other matrix");
    addOption(MATRIXINMEMORYCOLS, "c", "Number of cols of the other matrix");
    Map<String, List<String>> parsedArgs = parseArguments(strings);
    if (parsedArgs == null) {
      return -1;
    }

    Path aPath = new Path(getOption("atMatrix"));
    Path bPath = new Path(getOption("bMatrix"));
    int aCols = Integer.parseInt(getOption("numColsAt"));

    String inMemMatrixFileName = getOption(MATRIXINMEMORY);
    int inMemMatrixNumRows = Integer.parseInt(getOption(MATRIXINMEMORYROWS));
    int inMemMatrixNumCols = Integer.parseInt(getOption(MATRIXINMEMORYCOLS));

    run(getConf(), aPath, bPath, getOutputPath(), aCols, inMemMatrixFileName,
        inMemMatrixNumRows, inMemMatrixNumCols, 0f, 0f, 1);
    return 0;
  }

  /**
   * Refer to {@link CompositeDMJ} for further details.
   * 
   * @param conf the initial configuration
   * @param A transpose of matrix A
   * @param B matrix B
   * @param label the label for the output directory
   * @param numberOfJobs the hint for the desired number of parallel jobs
   * @return AxB wrapped in a DistributedRowMatrix object
   * @throws IOException
   * @throws InterruptedException
   * @throws ClassNotFoundException
   */
  public static DistributedRowMatrix run(Configuration conf,
      DistributedRowMatrix A, DistributedRowMatrix B, DistributedRowMatrix inMemC,   String label, float alpha1, float alpha2,
      int numberOfJobs) throws IOException, InterruptedException,
      ClassNotFoundException {
    log.info("running " + CompositeDMJ.class.getName());
    if (A.numRows() != B.numRows()) {
      throw new CardinalityException(A.numRows(), B.numRows());
    }
    if (A.numCols() != B.numCols()) {
      throw new CardinalityException(A.numCols(), B.numCols());
    }
    if (A.numCols() != inMemC.numRows()) {
      throw new CardinalityException(A.numCols(), inMemC.numRows());
    }
    if (inMemC.numCols() != inMemC.numRows()) {
      throw new CardinalityException(inMemC.numCols(), inMemC.numRows());
    }
    Path outPath = new Path(A.getOutputTempPath(), label);
    FileSystem fs = FileSystem.get(outPath.toUri(), conf);
    CompositeDMJ job = new CompositeDMJ();
    if (!fs.exists(outPath)) {
      job.run(conf, A.getRowPath(), B.getRowPath(), outPath, A.numRows(),
          inMemC.getRowPath(), inMemC.numRows(), inMemC.numCols(), alpha1, alpha2, numberOfJobs);
    } else {
      log.warn("----------- Skip already exists: " + outPath);
    }
    DistributedRowMatrix distRes =
        new DistributedRowMatrix(outPath, A.getOutputTempPath(), A.numRows(),
            A.numCols());
    distRes.setConf(conf);
    return distRes;
  }

  public void run(Configuration conf, Path aPath, Path bPath,
      Path matrixOutputPath, int atCols, Path inMemCDir, int inMemCRows,
      int inMemCCols, float alpha1, float alpha2, int numberOfJobs) throws IOException,
      InterruptedException, ClassNotFoundException {
    run(conf, aPath, bPath, matrixOutputPath, atCols, inMemCDir.toString(),
        inMemCRows, inMemCCols, alpha1, alpha2, numberOfJobs);
  }
  
  /**
   * Perform A x B, where A and B refer to the paths that contain matrices in
   * {@link SequenceFileInputFormat}. The smaller of At and B must also conform
   * with {@link MapDir} format. Refer to {@link CompositeDMJ} for further
   * details.
   * 
   * @param conf the initial configuration
   * @param aPath path to transpose of matrix A.
   * @param bPath path to matrix B
   * @param matrixOutputPath path to which AxB will be written
   * @param atCols number of columns of At (rows of A)
   * @param numberOfJobs the hint for the desired number of parallel jobs
   * @throws IOException
   * @throws InterruptedException
   * @throws ClassNotFoundException
   */
  public void run(Configuration conf, Path aPath, Path bPath,
      Path matrixOutputPath, int atCols,
      String inMemCStr, int inMemCRows, int inMemCCols, float alpha1, float alpha2,
      int numberOfJobs) throws IOException,
      InterruptedException, ClassNotFoundException {
    FileSystem fs = FileSystem.get(aPath.toUri(), conf);
    long atSize = MapDir.du(aPath, fs);
    long bSize = MapDir.du(bPath, fs);
    log.info("Choosing the smaller matrix: atSize: " + atSize + " bSize: "
        + bSize);
    boolean aIsMapDir = atSize < bSize;
    if (aIsMapDir)
      runJobsInParallel(conf, aPath, bPath, matrixOutputPath, atCols,
          aIsMapDir,
          inMemCStr, inMemCRows, inMemCCols, alpha1, alpha2, 
          numberOfJobs);
    else
      runJobsInParallel(conf, bPath, aPath, matrixOutputPath, atCols,
          aIsMapDir,
          inMemCStr, inMemCRows, inMemCCols, alpha1, alpha2, 
          numberOfJobs);
  }

  /**
   * Split a big job into multiple smaller jobs. Each job should be more
   * efficient as it puts less load on the reducers. We also can run the jobs in
   * parallel.
   * 
   * The maximum number of jobs is the number of column partitions produced by
   * the {@link ColPartitionedTransposeJob}.
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
      Path matrixInputPath, Path matrixOutputPath, int atCols,
      boolean aIsMapDir,
      String inMemCStr, int inMemCRows, int inMemCCols, float alpha1, float alpha2, 
      int numberOfJobs) throws IOException,
      InterruptedException, ClassNotFoundException {
    if (numberOfJobs == 1) {
      CompositeDMJ job = new CompositeDMJ();
      Job hjob =
          job.run(conf, mapDirPath, new Path[] { matrixInputPath },
              matrixOutputPath, atCols, aIsMapDir, 
              inMemCStr, inMemCRows, inMemCCols, alpha1, alpha2);
      boolean res = hjob.waitForCompletion(true);
      if (!res)
        throw new IOException("Job failed! ");
      return;
    }

    FileSystem fs = FileSystem.get(matrixOutputPath.toUri(), conf);
    FileStatus[] files =
        fs.listStatus(matrixInputPath, new AlgebraCommon.ExcludeMetaFilesFilter());
    int numColPartitions = computeNumOfColPartitions(files);

    // use the hint and max feasible parallelism to find an optimum degree for
    // parallelism
    numberOfJobs = Math.min(numColPartitions, numberOfJobs);
    int colPartitionsPerJob =
        (int) (Math.ceil(numColPartitions / (float) numberOfJobs));
    numberOfJobs =
        (int) (Math.ceil(numColPartitions / (float) colPartitionsPerJob));

    // run the jobs
    Job[] jobs = new Job[numberOfJobs];
    int nextPartitionIndex = 0;
    for (int jobIndex = 0; jobIndex < numberOfJobs; jobIndex++) {
      CompositeDMJ job = new CompositeDMJ();
      Path subJobOutPath = new Path(matrixOutputPath, "" + jobIndex);
      List<Path> inFilesList = new ArrayList<Path>(colPartitionsPerJob);
      int lastPartitionIndex =
          Math.min(nextPartitionIndex + colPartitionsPerJob, numColPartitions);
      for (; nextPartitionIndex < lastPartitionIndex; nextPartitionIndex++)
        addFilesOfAPartition(files, nextPartitionIndex, inFilesList);
      Path[] inFiles = new Path[inFilesList.size()];
      inFilesList.toArray(inFiles);
      jobs[jobIndex] =
          job.run(conf, mapDirPath, inFiles, subJobOutPath, atCols, aIsMapDir,
              inMemCStr, inMemCRows, inMemCCols, alpha1, alpha2);
    }

    // wait for the jobs (in case they are run in parallel and move their output
    // to the main output directory
    for (int jobIndex = 0; jobIndex < numberOfJobs; jobIndex++) {
      boolean res = jobs[jobIndex].waitForCompletion(true);
      if (!res)
        throw new IOException("Job failed! " + jobIndex);
      Path subJobDir = new Path(matrixOutputPath, "" + jobIndex);
      FileStatus[] jobOutFiles =
          fs.listStatus(subJobDir, new AlgebraCommon.ExcludeMetaFilesFilter());
      for (FileStatus jobOutFile : jobOutFiles) {
        Path src = jobOutFile.getPath();
        Path dst = new Path(matrixOutputPath, src.getName());
        // unique name by indexing with folder id
        log.info("fs.rename " + src + " -> " + dst);
        fs.rename(src, dst);
      }
      log.info("fs.delete " + subJobDir);
      fs.delete(subJobDir, true);
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
   * {@link MapDir} format. Refer to {@link CompositeDMJ} for further details.
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
      Path matrixOutputPath, int atCols, boolean aIsMapDir, 
      String inMemCStr, int inMemCRows, int inMemCCols, float alpha1, float alpha2) throws IOException,
      InterruptedException, ClassNotFoundException {
    conf.set(MATRIXINMEMORY, inMemCStr);
    conf.setInt(MATRIXINMEMORYROWS, inMemCRows);
    conf.setInt(MATRIXINMEMORYCOLS, inMemCCols);

    conf.setFloat(ALPHA1, alpha1);
    conf.setFloat(ALPHA2, alpha2);

    FileSystem fs = FileSystem.get(matrixOutputPath.toUri(), conf);
    NMFCommon.setNumberOfMapSlots(conf, fs, matrixInputPaths, "compositedmj");

    conf.set(MAPDIRMATRIX, mapDirPath.toString());
    conf.setBoolean(AISMAPDIR, aIsMapDir);
    @SuppressWarnings("deprecation")
    Job job = new Job(conf);
    job.setJarByClass(CompositeDMJ.class);
    job.setJobName(CompositeDMJ.class.getSimpleName() + "-" + matrixOutputPath.getName());
    matrixOutputPath = fs.makeQualified(matrixOutputPath);

    for (Path path : matrixInputPaths) {
      path = fs.makeQualified(path);
      MultipleInputs.addInputPath(job, path, SequenceFileInputFormat.class);
    }

    FileOutputFormat.setOutputPath(job, matrixOutputPath);
    job.setMapperClass(MyMapper.class);
    job.setMapOutputKeyClass(IntWritable.class);
    job.setMapOutputValueClass(VectorWritable.class);

    // ensures total order (when used with {@link MatrixOutputFormat}),
    RowPartitioner.setPartitioner(job, RowPartitioner.IntRowPartitioner.class,
        atCols);

    job.setNumReduceTasks(0);
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
    private VectorWritable otherVectorw = new VectorWritable();
    private VectorWritable outVectorw = new VectorWritable();
    private IntWritable outIntw = new IntWritable();
    private float alpha1;
    private float alpha2;
    
    private DenseMatrix inMemC;
    private DenseVector resVector = null;

    @Override
    public void setup(Context context) throws IOException {
      Configuration conf = context.getConfiguration();
      Path mapDirMatrixPath = new Path(conf.get(MAPDIRMATRIX));
      otherMapDir = new MapDir(conf, mapDirMatrixPath);
      aIsMapDir = conf.getBoolean(AISMAPDIR, true);
      alpha1 = conf.getFloat(ALPHA1, 0f);
      alpha2 = conf.getFloat(ALPHA2, 0f);
      
      Path inMemMatrixPath = new Path(conf.get(MATRIXINMEMORY));
      int inMemMatrixNumRows = conf.getInt(MATRIXINMEMORYROWS, 0);
      int inMemMatrixNumCols = conf.getInt(MATRIXINMEMORYCOLS, 0);
      inMemC = AlgebraCommon.mapDirToDenseMatrix(inMemMatrixPath,
          inMemMatrixNumRows, inMemMatrixNumCols, conf);
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
        multiplyWithInMem(mapDirVector);
      else
        multiplyWithInMem(normalInput);
      dotDivide(index, mapDirVector, normalInput, resVector, context);
      outVectorw.set(resVector);
      context.write(index, outVectorw);
    }
    
    public void multiplyWithInMem(Vector row)
        throws IOException, InterruptedException {
      if (resVector == null)
        resVector = new DenseVector(inMemC.numCols());
      AlgebraCommon.vectorTimesMatrix(row, inMemC, resVector);
      for (int i = 0; i < resVector.size(); i++) {
        double preVal = resVector.getQuick(i);
        double newVal = preVal + alpha2 * row.getQuick(i) + alpha1;
        resVector.setQuick(i, newVal);
      }
    }

    //TODO: what if the vector is sparse?
    void dotDivide(IntWritable index, Vector aVector, Vector bVector, Vector cVector, Context context)
        throws IOException, InterruptedException {
      for (int i = 0; i < aVector.size(); i++) {
        double ai = aVector.getQuick(i); 
        double ci = cVector.getQuick(i);
        double res = 0;
        if (ci != 0)
          res = ai * bVector.getQuick(i) / ci;
        else if (ai != 0)
          context.getCounter("Error", "NaN").increment(1);

        resVector.setQuick(i, res);
      }
    }

    @Override
    public void cleanup(Context context) throws IOException {
      otherMapDir.close();
    }
  }
}
