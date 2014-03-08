package com.myabandeh.algebra.matrix.format;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Iterator;
import java.util.Map;
import java.util.Scanner;
import java.util.TreeMap;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.FileUtil;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.PathFilter;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.MapFile;
import org.apache.mahout.math.MatrixSlice;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.hadoop.DistributedRowMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * A utility class for random read from a directory of map files. For the first
 * random read, this class identifies the MapFile which contains the key. After
 * the first random read, the subsequent requested keys are expected to be close
 * to the last read key, so that it allows efficient sequential reading from the
 * map file. This class automatically switches to the next MapFile when the map
 * file finishes.
 * 
 * Assumes: (1) the title of each MapFile is tagged with the smallest key in the
 * map file, (2) the range of keys covered by each MapFile are disjoint. These
 * preconditions are automatically preserved if the MapDir is generated by a job
 * that has the following properties:
 * 
 * (a) the job output format is set to {@link MatrixOutputFormat}
 * 
 * (b) the job partitioner is derived from {@link RowPartitioner}
 */
public class MapDir {
  private static final Logger log = LoggerFactory.getLogger(MapDir.class);

  /**
   * Maps keys to MapFiles that contain them.
   */
  private TitlePartitioner partitioner;

  private class Readers {
    private MapFile.Reader readers[];
    private Path[] names;
    private FileSystem fs;
    private Configuration conf;

    private Readers(Path dir, Configuration conf) throws IOException {
      this.conf = conf;
      fs = dir.getFileSystem(conf);
      names = FileUtil.stat2Paths(fs.listStatus(dir, mapFilter));
      readers= new MapFile.Reader[names.length];

      //TODO ensure only one instance of Readers
      partitioner = new TitlePartitioner(names);
    }

    public MapFile.Reader getReader(int i) throws IOException {
      if (readers[i] == null)
        readers[i] = new MapFile.Reader(fs, names[i].toString(), conf);
      return readers[i];
    }

    public void close() throws IOException {
      for (MapFile.Reader reader: readers) {
        if (reader != null)
          reader.close();
      }
      readers = null;
    }
  }
  
  Readers readers;
  
  /**
   * The reader to the last used MapFile
   */
  private MapFile.Reader lastReader = null;

  /**
   * The next key in sequence. It is valid only if lastReader is not null.
   */
  private IntWritable nextKey = new IntWritable();

  /**
   * The next value in sequence. It is valid only if lastReader is not null.
   */
  private VectorWritable nextValue = new VectorWritable();

  private boolean noMorePartitions = false;
  
  /**
   * Create a new MapDir
   * @param conf
   * @param inPath the directory containing the MapFiles
   * @throws IOException
   */
  public MapDir(Configuration conf, Path inPath) throws IOException {
    readers = new Readers(inPath, conf);
  }

  /**
   * [Re-]Initialize the MapDir with new directory of MapFiles. {@link #close()}
   * must be called before if MapDir is already initialized.
   * 
   * @param dir
   * @param conf
   * @return
   * @throws IOException
   */
  @SuppressWarnings("deprecation")
  public MapFile.Reader[] getReaders(Path dir, Configuration conf)
      throws IOException {
    FileSystem fs = dir.getFileSystem(conf);
    Path[] names = FileUtil.stat2Paths(fs.listStatus(dir, mapFilter));

    partitioner = new TitlePartitioner(names);

    MapFile.Reader[] parts = new MapFile.Reader[names.length];
    for (int i = 0; i < names.length; i++) {
      parts[i] = new MapFile.Reader(fs, names[i].toString(), conf);
    }
    return parts;
  }

  /**
   * Close the MapFiles and release the resources. It also allows the MapDir to be
   * reinitialized.
   * @throws IOException
   */
  public void close() throws IOException {
    readers.close();
  }
  
  /**
   * Jump to the reader that contains the key
   * 
   * @param key
   * @throws IOException
   */
  private MapFile.Reader loadReader(IntWritable key) throws IOException {
    int partitionIndex = partitioner.getPartitionIndex(key.get());
    if (partitionIndex == partitioner.getLastPartitionIndex())
      noMorePartitions = true;
    log.info("Partition index is " + partitionIndex + " key was: " + key.get());
    lastReader = readers.getReader(partitionIndex);
    return lastReader;
  }

  /**
   * Get the value associated with the key
   * @param key
   * @param val the object that will be filled with the retrieved value 
   * @return the retrieved value
   * @throws IOException
   */
  public VectorWritable get(IntWritable key, VectorWritable val)
      throws IOException {
    if (lastReader == null && noMorePartitions)
      return null;
    if (lastReader == null) {
      loadReader(key);
      nextKey.set(key.get());
      boolean eof = lastReader.getClosest(nextKey, nextValue, true) == null;
      if (eof) {
        lastReader = null;
        return null;
      }
    }
    boolean eof = false;
    //skip over keys until find the one that the user is asking for. This should rarely 
    //occur as the user normally asks for sequential keys
    while (!eof && nextKey.compareTo(key) < 0)
      eof = !lastReader.next(nextKey, nextValue);
    //If the requested key is not in the current MapFile, reset the process and 
    //search in the next MapFile using recursive call
    if (eof) {
      lastReader = null;
      return get(key, val);
    }
    if (nextKey.equals(key)) {
      val.set(nextValue.get());
      //update nextKey and nextValue for the next call
      eof = !lastReader.next(nextKey, nextValue);
      if (eof)
        lastReader = null;
      return val;
    }
    return null;
  }

  /**
   * Iterate over all entries. The behavior is undefined if it is interleaved with 
   * normal {@link #get(IntWritable, VectorWritable)} method.
   * @return
   * @throws IOException
   */
  public Iterator<MatrixSlice> iterateAll() throws IOException {
    return new MapDirIterator();
  }

  public class MapDirIterator implements Iterator<MatrixSlice> {
    int lastReaderFirstKey;
    private boolean eof;

    public MapDirIterator() throws IOException {
      lastReaderFirstKey = partitioner.getFirstKey();
      int lastReaderIndex = partitioner.getPartitionIndex(lastReaderFirstKey);
      lastReader = readers.getReader(lastReaderIndex);
      eof = false;
      readNext();
    }
    
    private void readNext() throws IOException {
      if (eof)
        return;
      eof = !lastReader.next(nextKey, nextValue);
      while (eof) {
        lastReaderFirstKey = partitioner.getNextPartitionFirstKey(lastReaderFirstKey);
        if (lastReaderFirstKey == Integer.MIN_VALUE)
          return;
        int lastReaderIndex = partitioner.getPartitionIndex(lastReaderFirstKey);
        lastReader = readers.getReader(lastReaderIndex);
        eof = !lastReader.next(nextKey, nextValue);
      }
    }
    
    @Override
    public boolean hasNext() {
      return !eof;
    }

    @Override
    public MatrixSlice next() {
      MatrixSlice slice = new MatrixSlice(nextValue.get(), nextKey.get());
      try {
        //for the next round
        readNext();
      } catch (IOException e) {
        log.error(e.getMessage());
        e.printStackTrace();
        eof = true;
        return null;
      }
      return slice;
    }

    @Override
    public void remove() {
      // TODO Auto-generated method stub
    }
    
  }

  public static void main(String[] args) throws Exception {
    Path inPath = new Path(args[0]);
    Configuration conf = new Configuration();
    MapDir mapDir = new MapDir(conf, inPath);
    for (int i = 0; i < 10; i++) {
      IntWritable key = new IntWritable(i);
      VectorWritable vw = new VectorWritable();
      vw = mapDir.get(key, vw);
      System.out.println(vw);
    }
    mapDir.close();
  }

  public static void testIterator(DistributedRowMatrix origMtx, Path inPath)
      throws IOException {
    Configuration conf = new Configuration();
    MapDir mapDir = new MapDir(conf, inPath);

    Iterator<MatrixSlice> sliceIterator = origMtx.iterateAll();
    while (sliceIterator.hasNext()) {
      MatrixSlice slice = sliceIterator.next();
      int index = slice.index();
      System.out.println("A[" + index + "] = " + slice.vector());

      IntWritable key = new IntWritable(index);
      VectorWritable vw = new VectorWritable();
      vw = mapDir.get(key, vw);
      System.out.println("B[" + index + "] = " + vw);
    }
    mapDir.close();
  }

  /**
   * Split the range of keys between the map files, based on the smallest key in
   * the map file extracted from its title. The key could be obtained by reading
   * the map file as well, but then the entire block has to be loaded which is
   * Inefficient when it is done for each MapFile from every reader node. Map
   * keys to the MapFile that contains them based on the title of the map file
   * 
   * @author myabandeh
   */
  static class TitlePartitioner {
    TreeMap<Integer, Integer> partitionmap = new TreeMap<Integer, Integer>();

    public TitlePartitioner(Path[] paths) {
      for (int i = 0; i < paths.length; i++) {
        Integer key = extractKeyFromTitle(paths[i]);
        partitionmap.put(key, i);
//        log.info("partitionmap: key: " + key + " i: " + i + " paths[i]: " + paths[i]);
      }
    }

    public int getPartitionIndex(int key) {
      Map.Entry<Integer, Integer> entry = partitionmap.floorEntry(key);
      if (entry == null)
        entry = partitionmap.firstEntry();
      return entry.getValue();
    }
    
    public int getLastPartitionIndex() {
      return partitionmap.lastEntry().getValue();
    }
    
    public int getFirstKey() {
      return partitionmap.firstKey();
    }

    /**
     * What is the first key of the next partition?
     * 
     * @param key
     *          the first key of the current partition
     * @return the first key of the next partition or Integer.MIN_VALUE if there
     *         is no next partition
     */
    public int getNextPartitionFirstKey(int key) {
      Map.Entry<Integer, Integer> entry = partitionmap.ceilingEntry(key+1);
      if (entry == null)
        return Integer.MIN_VALUE;
      return entry.getKey();
    }
  }

  static int extractKeyFromTitle(Path path) {
    String name = path.getName();// matrix-k-123
    name = name.replace("--","-");//TODO: there is a bug that inserts two -
    Scanner scanner = new Scanner(name);
    scanner.useDelimiter("-");
    scanner.next();// matrix
    scanner.next();// k
    int key = scanner.nextInt();
    return key;
  }

  static PathFilter mapFilter = new PathFilter() {
    @Override
    public boolean accept(Path path) {
      String name = path.getName();
      // if (name.startsWith("_") || name.startsWith("."))
      // return false;
      if (name.startsWith("matrix-k-"))
        return true;
      return false;
    }
  };
  
  /**
   * Disk usage of the MapDir or a dir of sequence files
   * @param mapDirPath the path to MapDir or a directory of sequence files
   * @param fs
   * @return
   * @throws FileNotFoundException
   * @throws IOException
   */
  public static long du(Path mapDirPath, FileSystem fs) throws FileNotFoundException,
      IOException {
    FileStatus[] dirs = fs.listStatus(mapDirPath, mapFilter);
    if (dirs.length == 0) //it is not a mapdir then, do a simple ls
      dirs = fs.listStatus(mapDirPath);
    long size = 0;
    for (FileStatus dirStatus : dirs) {
      //if it is a sequence file
      size += dirStatus.getLen();
      //or if it is a mapfile, which is directory
      size += dirSize(dirStatus, fs);
    }
    return size;
  }

  //each directory is composed of files, and zero size directories
  private static long dirSize(FileStatus dirStatus, FileSystem fs)
      throws FileNotFoundException, IOException {
    FileStatus[] files = fs.listStatus(dirStatus.getPath(), new PathFilter() {
      @Override
      public boolean accept(final Path file) {
        return true;
      }
    });
    long size = 0;
    for (FileStatus file : files)
      size += file.getLen();
    return size;
  }  

}
