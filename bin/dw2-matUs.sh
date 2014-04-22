s=`date`
LOG=dw2$1.log
hadoop jar algebra-1.0-SNAPSHOT-job.jar com.twitter.algebra.nmf.NMFDriver \
	-Dmapreduce.job.split.metainfo.maxsize=-1 \
	-Dyarn.app.mapreduce.am.resource.mb=8096 \
	-Dyarn.app.mapreduce.am.command-opts='-Xmx7000m' \
	-Dmapred.job.map.memory.mb=3072 \
	-Dmapreduce.task.io.sort.mb=1 \
	-Dalgebra.mapslots=2000 \
	-Dalgebra.reduceslots=2000 \
	-Dalgebra.reduceslots.transpose=200 \
	-Dalgebra.reduceslots.xtx=200 \
	-Dalgebra.reduceslots.combiner=1 \
	-Dmapreduce.map.java.opts='-Xmx2800m' \
	-Dmapred.task.timeout=9000000 \
	-Dmatrix.atb.epsilon='1e-10' \
	-Dmatrix.text.epsilon='1e-10' \
	-Dmatrix.nmf.composite.epsilon='1e-10' \
	-Dmatrix.nmf.stop.rounds=25 \
	-i matinput/Format-us -o matoutput --tempDir mattmpUSsparse -rows 37200000 -cols 37200000 -pcs 1000 -parts 500 2>&1 | tee $LOG
e=`date`
echo $s
echo $e
echo $s >> $LOG
echo $e >> $LOG

cp $LOG ${LOG}.`date +%s`

	#-Dmapred.job.reduce.memory.mb=2048 \
	#-Dmapreduce.reduce.java.opts='-Xmx2048m' \
	#-Dmapreduce.task.io.sort.mb=1 \
. ./othercmd.sh
