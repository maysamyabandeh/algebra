package com.twitter.algebra.nmf;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.HashMap;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.SequenceFile;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.twitter.algebra.matrix.format.MapDir;

public class NMFCommon {
  private static final Logger log = LoggerFactory.getLogger(NMFCommon.class);

  static final String MAPSPLOTS = "algebra.mapslots";
  static final String REDUCESLOTS = "algebra.reduceslots";
  public static int DEFAULT_REDUCESPLOTS = 100;

  public static void main(String[] args) throws IOException {
    if (args.length < 1) {
      System.err.println("input text file is missing");
      return;
    }
    readHashMap(args[0]);
  }

  public static HashMap<Long, Integer> readHashMap(String inputStr)
      throws IOException {
    HashMap<Long, Integer> hashMap = new HashMap<Long, Integer>();

    Configuration conf = new Configuration();
    Path finalNumberFile = new Path(inputStr + "/part-r-00000");
    @SuppressWarnings("deprecation")
    SequenceFile.Reader reader =
        new SequenceFile.Reader(FileSystem.get(conf), finalNumberFile, conf);
    double sum = 0;
    LongWritable key = new LongWritable();
    IntWritable value = new IntWritable();
    while (reader.next(key, value)) {
      hashMap.put(key.get(), value.get());
    }
    System.out.println("SUM " + sum);
    reader.close();
    return hashMap;
  }

  public static void setNumberOfMapSlots(Configuration conf, FileSystem fs,
      Path path, String joblabel) {
    setNumberOfMapSlots(conf, fs, new Path[] {path}, joblabel);
  }

  public static void setNumberOfMapSlots(Configuration conf, FileSystem fs,
      Path[] paths, String joblabel) {
    if (conf.get(MAPSPLOTS) == null)
      return;
    int mapSlots = conf.getInt(MAPSPLOTS, 1);
    mapSlots = conf.getInt(MAPSPLOTS + "." + joblabel, mapSlots);
    long du = 0;
    try {
      for (Path path : paths)
        du += MapDir.du(path, fs);
    } catch (FileNotFoundException e) {
      e.printStackTrace();
    } catch (IOException e) {
      e.printStackTrace();
    }
    long splitSize = du / mapSlots;
    log.info("du: " + du + " mapSlots: " + mapSlots + " splitSize: " + splitSize);
    long minSplitSize = (long) (splitSize * 0.9);
    long maxSplitSize = Math.max((long) (splitSize * 1.1), 1024 * 1024);
    conf.setLong("mapred.min.split.size", minSplitSize);
    conf.setLong("mapreduce.min.split.size", minSplitSize);
    conf.setLong("mapred.max.split.size", maxSplitSize);
    conf.setLong("mapreduce.max.split.size", maxSplitSize);
  }

  public static int getNumberOfReduceSlots(Configuration conf, String joblabel) {
    int redSlots = conf.getInt(REDUCESLOTS, DEFAULT_REDUCESPLOTS);
    redSlots = conf.getInt(REDUCESLOTS + "." + joblabel, redSlots);
    return redSlots;
  }

  static void printMemUsage() {
    int mb = 1024 * 1024;
    // Getting the runtime reference from system
    Runtime runtime = Runtime.getRuntime();
    System.out.println("##### Heap utilization statistics [MB] #####");
    // Print used memory
    System.out.print("Used Memory:"
        + (runtime.totalMemory() - runtime.freeMemory()) / mb);
    // Print free memory
    System.out.print(" Free Memory:" + runtime.freeMemory() / mb);
    // Print total available memory
    System.out.print(" Total Memory:" + runtime.totalMemory() / mb);
    // Print Maximum available memory
    System.out.print(" Max Memory:" + runtime.maxMemory() / mb);
  }

  public static int computeOptColPartitionsForMemCombiner(Configuration conf,
      int rows, int cols) {
    final int MB = 1024 * 1024;
    final int MEMBYTES = conf.getInt("mapreduce.map.memory.mb", 1024);
    int availableMem = (MEMBYTES - 512 /* jvm */) / 2; //use only half for combiner
    int colParts = (int) (rows / (float) availableMem / MB * cols * 8); /*bytes per double element*/
    return colParts;
  }
}
