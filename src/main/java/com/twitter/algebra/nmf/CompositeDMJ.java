package com.twitter.algebra.nmf;

import java.io.IOException;
import java.util.List;
import java.util.Map;

import org.apache.hadoop.conf.Configuration;
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
import org.apache.mahout.math.SequentialAccessSparseVector;
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
 * (A ./ (A*MEM+A.*a2+a1)) .* B:
 * 
 * Approach: Broadcast of MEM and Dynamic Mapside join of B
 */
public class CompositeDMJ extends AbstractJob {
  private static final Logger log = LoggerFactory.getLogger(CompositeDMJ.class);

  public static final String MAPDIRMATRIX = "CompostiteDMJ.MapDirMatrix";

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
    addOption("aMatrix", "aMatrix", "The first matrix");
    addOption("bMatrix", "bMatrix", "The second matrix");
    addOption(MATRIXINMEMORY, "times",
        "The name of the file that contains the matrix that fits into memory");
    addOption(MATRIXINMEMORYROWS, "r", "Number of rows of the other matrix");
    addOption(MATRIXINMEMORYCOLS, "c", "Number of cols of the other matrix");
    Map<String, List<String>> parsedArgs = parseArguments(strings);
    if (parsedArgs == null) {
      return -1;
    }

    Path aPath = new Path(getOption("aMatrix"));
    Path bPath = new Path(getOption("bMatrix"));
    int aCols = Integer.parseInt(getOption("numColsA"));

    String inMemMatrixFileName = getOption(MATRIXINMEMORY);
    int inMemMatrixNumRows = Integer.parseInt(getOption(MATRIXINMEMORYROWS));
    int inMemMatrixNumCols = Integer.parseInt(getOption(MATRIXINMEMORYCOLS));

    run(getConf(), aPath, bPath, getOutputPath(), aCols, inMemMatrixFileName,
        inMemMatrixNumRows, inMemMatrixNumCols, 0f, 0f);
    return 0;
  }

  /**
   * Refer to {@link CompositeDMJ} for further details.
   */
  public static DistributedRowMatrix run(Configuration conf,
      DistributedRowMatrix A, DistributedRowMatrix B,
      DistributedRowMatrix inMemC, String label, float alpha1, float alpha2)
      throws IOException, InterruptedException, ClassNotFoundException {
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
          inMemC.getRowPath(), inMemC.numRows(), inMemC.numCols(), alpha1,
          alpha2);
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
      int inMemCCols, float alpha1, float alpha2) throws IOException,
      InterruptedException, ClassNotFoundException {
    run(conf, aPath, bPath, matrixOutputPath, atCols, inMemCDir.toString(),
        inMemCRows, inMemCCols, alpha1, alpha2);
  }

  public void run(Configuration conf, Path aPath, Path bPath,
      Path matrixOutputPath, int atCols, String inMemCStr, int inMemCRows,
      int inMemCCols, float alpha1, float alpha2) throws IOException,
      InterruptedException, ClassNotFoundException {
    FileSystem fs = FileSystem.get(aPath.toUri(), conf);
    long atSize = MapDir.du(aPath, fs);
    long bSize = MapDir.du(bPath, fs);
    log.info("Choosing the smaller matrix: atSize: " + atSize + " bSize: "
        + bSize);
    boolean aIsMapDir = atSize < bSize;
    CompositeDMJ job = new CompositeDMJ();
    Job hjob;
    if (aIsMapDir)
      hjob =
          job.run(conf, aPath, bPath, matrixOutputPath, atCols, aIsMapDir,
              inMemCStr, inMemCRows, inMemCCols, alpha1, alpha2);
    else
      hjob =
          job.run(conf, bPath, aPath, matrixOutputPath, atCols, aIsMapDir,
              inMemCStr, inMemCRows, inMemCCols, alpha1, alpha2);
    boolean res = hjob.waitForCompletion(true);
    if (!res)
      throw new IOException("Job failed! ");
  }

  public Job run(Configuration conf, Path mapDirPath, Path matrixInputPaths,
      Path matrixOutputPath, int atCols, boolean aIsMapDir, String inMemCStr,
      int inMemCRows, int inMemCCols, float alpha1, float alpha2)
      throws IOException, InterruptedException, ClassNotFoundException {
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
    job.setJobName(CompositeDMJ.class.getSimpleName() + "-"
        + matrixOutputPath.getName());
    matrixOutputPath = fs.makeQualified(matrixOutputPath);

    matrixInputPaths = fs.makeQualified(matrixInputPaths);
    MultipleInputs.addInputPath(job, matrixInputPaths,
        SequenceFileInputFormat.class);

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

  public static class MyMapper extends
      Mapper<IntWritable, VectorWritable, IntWritable, VectorWritable> {
    private MapDir otherMapDir;
    private boolean aIsMapDir;
    private VectorWritable otherVectorw = new VectorWritable();
    private VectorWritable outVectorw = new VectorWritable();
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
      inMemC =
          AlgebraCommon.mapDirToDenseMatrix(inMemMatrixPath,
              inMemMatrixNumRows, inMemMatrixNumCols, conf);
      resVector = new DenseVector(inMemC.numCols());
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
        composite(mapDirVector, normalInput, context);
      else
        composite(normalInput, mapDirVector, context);
      outVectorw.set(new SequentialAccessSparseVector(resVector));
      context.write(index, outVectorw);
    }

    static final double EPSLION = 1e-10;

    // a . b / c
    private void composite(Vector aVector, Vector bVector, Context context) {
      for (int i = 0; i < aVector.size(); i++) {
        double ai = aVector.getQuick(i);
        if (ai ==0) {
          resVector.setQuick(i, 0);
          continue;
        }
        double bi = bVector.getQuick(i);
        if (bi ==0) {
          resVector.setQuick(i, 0);
          continue;
        }
        
        Double preVal = aVector.dot(inMemC.viewColumn(i));
        double ci = preVal + alpha2 * ai + alpha1;

        double res = 0;
        if (ci != 0)
          res = ai * bi / ci;
        else if (ai != 0)
          context.getCounter("Error", "NaN").increment(1);
        
        if (res < EPSLION)
          res = 0;

        resVector.setQuick(i, res);
      }
    }

    public void multiplyWithInMem(Vector row) throws IOException,
        InterruptedException {
      AlgebraCommon.vectorTimesMatrix(row, inMemC, resVector);
      for (int i = 0; i < resVector.size(); i++) {
        double preVal = resVector.getQuick(i);
        double newVal = preVal + alpha2 * row.getQuick(i) + alpha1;
        resVector.setQuick(i, newVal);
      }
    }

    // TODO: what if the vector is sparse?
    void dotDivideByResVector(Vector aVector, Vector bVector, Context context)
        throws IOException, InterruptedException {
      // I do a full iteration here since I have to zero the un-initializied
      // resVector
      for (int i = 0; i < aVector.size(); i++) {
        double ai = aVector.getQuick(i);
        double bi = bVector.getQuick(i);
        double ci = resVector.getQuick(i);
        double res = 0;
        if (ci != 0)
          res = ai * bi / ci;
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
