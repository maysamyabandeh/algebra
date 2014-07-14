/*
Copyright 2014 Twitter, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package com.twitter.algebra.matrix.multiply;

import java.io.IOException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.common.commandline.DefaultOptionCreator;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.hadoop.DistributedRowMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.twitter.algebra.AlgebraCommon;
import com.twitter.algebra.TransposeJob;
import com.twitter.algebra.matrix.format.Sequence2MatrixFormatJob;
import com.twitter.algebra.matrix.multiply.AtB_DMJ;

/**
 * Benchmark multiplication algorithms
 * 
 * @author myabandeh
 * 
 */
public class MultiplicationDriver extends AbstractJob {
  private static final Logger log = LoggerFactory.getLogger(MultiplicationDriver.class);

  private static final String BPATH = "B";
  private static final String ROWSOPTION = "N";
  private static final String COLSOPTION = "D";
  private static final String PRINCIPALSOPTION = "d";
  private static final String PARTITIONSOPTION = "kp";

  @Override
  public int run(String[] args) throws Exception {
    addInputOption();
    addOutputOption();
    addOption(DefaultOptionCreator.methodOption().create());
    addOption(BPATH, "bPath", "path to matrix B");
    addOption(ROWSOPTION, "rows", "Number of rows");
    addOption(COLSOPTION, "cols", "Number of cols");
    addOption(PRINCIPALSOPTION, "pcs", "Number of principal components");
    addOption(PARTITIONSOPTION, "parts", "Number of partitions in principal components");
    if (parseArguments(args) == null) {
      return -1;
    }
    Path input = getInputPath();
    Path output = getOutputPath();
    final Path bPath = new Path(getOption(BPATH));
    final int nRows = Integer.parseInt(getOption(ROWSOPTION));
    final int nCols = Integer.parseInt(getOption(COLSOPTION));
    final int nPCs = Integer.parseInt(getOption(PRINCIPALSOPTION));
    final int nColPartitions = Integer.parseInt(getOption(PARTITIONSOPTION));


    Configuration conf = getConf();
    if (conf == null) {
      throw new IOException("No Hadoop configuration present");
    }
    
    run(conf, input, bPath, output, nRows, nCols, nPCs, nColPartitions);
    return 0;
  }

  private void run(Configuration conf, Path atPath, Path bPath, Path output, int nRows,
      int nCols, int k, int nParts) throws Exception {
    log.info("reading At");
    DistributedRowMatrix distAt =
        new DistributedRowMatrix(atPath, getTempPath(), nRows, nCols);
    distAt.setConf(conf);

    log.info("reading B");
    DistributedRowMatrix distB =
        new DistributedRowMatrix(bPath, getTempPath(), nRows, k);
    distB.setConf(conf);

    log.info("Partitioning At");
    distAt = PartitionerJob.run(conf, distAt, nParts, atPath.getName() + "-partitioned" + nParts);
    log.info("Partitioning B");
    distB = PartitionerJob.run(conf, distB, nParts, bPath.getName() + "-partitioned" + nParts);
    
    log.info("Computing At x B");
    DistributedRowMatrix distXt = 
        AtBOuterStaticMapsideJoinJob.run(conf, distAt, distB, atPath.getName() + "x" + bPath.getName() + "-SMJ");
  }

  /**
   * @param args
   * @throws Exception
   */
  public static void main(String[] args) throws Exception {
    ToolRunner.run(new Configuration(), new MultiplicationDriver(), args);
  }

}
