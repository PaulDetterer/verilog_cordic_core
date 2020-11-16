#### Template script for rtlstim2gate flow, generated from Joules(TM) RTL Power Solution, Version v19.11-s017_1 (Aug 26 2019 21:54:52)
if { [file exists /proc/cpuinfo] } {
    sh grep "model name" /proc/cpuinfo
    sh grep "cpu MHz"    /proc/cpuinfo
}

puts "Hostname : [info hostname]"

##############################################################################
## Preset global variables and attributes
##############################################################################
set DESIGN cordic
set SYN_EFF medium

set MAP_EFF low
set today  [cdn_get_date]

set d_outputs $joulesWorkDir/outputs/$today
set d_reports $joulesWorkDir/reports/$today
set d_logs    $joulesWorkDir/logs/$today

foreach dir "logs reports outputs" { cdn_mkdir $joulesWorkDir/$dir; cdn_mkdir $joulesWorkDir/$dir/$today }

#::legacy::set_attribute init_lib_search_path {. ./lib} / 
set_attribute init_lib_search_path {. ./lib \
    /opt/tsmc/40nm/TSMCHOME/digital/Front_End/timing_power_noise/ECSM/tcbn40lpbwp_120b \
    /opt/tsmc/40nm/TSMCHOME/memories/Front_End/ts1n40lpb256x32m4m_210b/NLDM} /
::legacy::set_attribute script_search_path {./scr} /
::legacy::set_attribute init_hdl_search_path {../rtl} /

::legacy::set_attribute information_level 9 /
applet load report_histogram

###############################################################
## Library setup
###############################################################
#read_libs <list_of_library_names>
read_libs tcbn40lpbwpwcz_ecsm.lib

####################################################################
## tag a cell as memory 
####################################################################
#tag_memory -cell <glob_memory_cell>

####################################################################
## Load Design
####################################################################

if {$env(CFG)=="comb"} {
    read_hdl -define COMBINATORIAL ../rtl/cordic.v
} elseif {$env(CFG)=="fpl"} {
    read_hdl -define PIPELINE ../rtl/cordic.v
}

elaborate $DESIGN
puts "Runtime & Memory after 'read_hdl'"
timestat Elaboration

check_design -unresolved


################################################################################################
## read in stimulus file.
################################################################################################
set stimFile ../sim/activity/$env(CFG).tcf
read_stimulus -file $stimFile 

#read_stimulus -file <stimulus_file_name_1> -dut_instance <dut_instance_name> 
#read_stimulus -file <stimulus_file_name_2> -dut_instance <dut_instance_name> -append

################################################################################################
## write SDB.
################################################################################################
write_sdb -out $d_outputs/$env(CFG).sdb
#suspend

####################################################################
## constraints setup
####################################################################
#read_sdc <sdc_file_name>

if {$env(CFG) == "comb"} {
    path_delay -delay $env(CP) -from [all_inputs] -to [all_outputs]
} elseif {$env(CFG) == "fpl"} {
    define_clock -name CLK -period $env(CP) [find / -port clk]
    path_disable -from [find -port rst]
}



# Turn on TNS, affects global and incr opto (doesn't do much w/ global map)
if {$env(CFG) != "comb"} {
    ::legacy::set_attribute lp_insert_clock_gating true /
    ::legacy::set_attribute lp_clock_gating_min_flops  3 $DESIGN
    ::legacy::set_attribute lp_clock_gating_max_flops  8 $DESIGN 
}

################################################################################
# synthesize to gates
################################################################################

power_map -effort $MAP_EFF 
puts "Runtime & Memory after 'power_map'"
timestat MAPPED
write_db -all -to_file $joulesWorkDir/proto.db
read_stimulus -file $d_outputs/$env(CFG).sdb


compute_power -mode average
report_activity  -out $d_reports/$env(CFG).activity.rep
report_power -by_hier -out $d_reports/$env(CFG).power.rep
report_icgc_efficiency -out $d_reports/$env(CFG).icgc.rep
report area > $d_reports/$env(CFG).area.rep
puts "Final Runtime & Memory."
timestat FINAL

##quit
