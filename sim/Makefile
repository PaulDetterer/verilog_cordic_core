

CP=31.250
comb.tcf : 
	export CFG=comb; xrun -define COMBINATORIAL -define CP=${CP} -f xrun.args -input scr/dumpAndRun.tcl

fpl.tcf: 
	export CFG=fpl; xrun -define PIPELINE -define CP=${CP} -f xrun.args -input scr/dumpAndRun.tcl 
fpl_gui: 
	export CFG=fpl; xrun -define PIPELINE -define CP=${CP} -f xrun.args -gui
