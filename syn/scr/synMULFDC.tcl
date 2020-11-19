#### Template Script for RTL->Gate-Level Flow (generated from GENUS 19.11-s087_1) 

if {[file exists /proc/cpuinfo]} {
  sh grep "model name" /proc/cpuinfo
  sh grep "cpu MHz"    /proc/cpuinfo
}

puts "Hostname : [info hostname]"

##############################################################################
## Preset global variables and attributes
##############################################################################


set DESIGN MULFDC
set GEN_EFF medium
set MAP_OPT_EFF high
set DATE [clock format [clock seconds] -format "%b%d-%T"] 
set _OUTPUTS_PATH outputs_${DATE}
set _REPORTS_PATH reports_${DATE}
set _LOG_PATH logs_${DATE}
##set ET_WORKDIR <ET work directory>
set_db / .init_lib_search_path {. ./lib \
    /opt/tsmc/40nm/TSMCHOME/digital/Front_End/timing_power_noise/ECSM/tcbn40lpbwp_120b \
    /opt/tsmc/40nm/TSMCHOME/memories/Front_End/ts1n40lpb256x32m4m_210b/NLDM} 
set_db / .script_search_path {. ./scr} 
set_db / .init_hdl_search_path {. ../rtl} 
##Uncomment and specify machine names to enable super-threading.
##set_db / .super_thread_servers {<machine names>} 
##For design size of 1.5M - 5M gates, use 8 to 16 CPUs. For designs > 5M gates, use 16 to 32 CPUs
##set_db / .max_cpus_per_server 8

##Default undriven/unconnected setting is 'none'.  
##set_db / .hdl_unconnected_value 0 | 1 | x | none

set_db / .information_level 7 

###############################################################
## Library setup
###############################################################


#read_libs <libname>
read_libs {tcbn40lpbwpwcz_ecsm.lib ts1n40lpb256x32m4m_210b_ss0p99v0c.lib}
#read_physical -lef <lef file(s)>
read_physical -lef {/opt/tsmc/40nm/TSMCHOME/digital/Back_End/lef/tcbn40lpbwp_120c/lef/HVH_0d5_0/tcbn40lpbwp_8lm5X2ZRDL.lef \
                            /opt/tsmc/40nm/TSMCHOME/memories/Front_End/ts1n40lpb256x32m4m_210b/LEF/ts1n40lpb256x32m4m_210b_4m.lef}

read_qrc /opt/tsmc/40nm/TSMCHOME/digital/util/RC_QRC_cln40lp_1p08m+alrdl_5x2z/typical/qrcTechFile
## Provide either cap_table_file or the qrc_tech_file
#set_db / .cap_table_file <file> 
#read_qrc <qrcTechFile name>

set_db / .lp_insert_clock_gating true 

####################################################################
## Load Design
####################################################################
read_hdl fir.v
read_hdl -sv shifter.sv
read_hdl -sv MULFDC.sv
elaborate -parameters {{BW 12} {ABW 10}} $DESIGN
set DESIGN ${DESIGN}_BW12_ABW10
puts "Runtime & Memory after 'read_hdl'"
time_info Elaboration



check_design -unresolved

####################################################################
## Constraints Setup
####################################################################


# 32 MHz clock 
#define_clock -period 31250  -name CLK [vfind -port clk]
#external_delay -input <delay value>  -clock <object> 
#external_delay -output <delay value>  -clock <object>
#set_db <port> .external_driver <libcell pin>
#set_db <port> .driver_input_slew_fall_to_rise_max <integer> 
#set_db <port> .driver_input_slew_fall_to_rise_min <integer> 
#set_db <port> .driver_input_slew_fall_to_fall_max <integer> 
#set_db <port> .driver_input_slew_fall_to_fall_min <integer> 
#set_db "design:$DESIGN" .max_fanout <value> 
#set_db "design:$DESIGN" .max_capacitance <value in fF>
#set_db "design:$DESIGN" .max_transition <value in ps> 
#path_disable -from [vfind -port rst] 
#multi_cycle -from <object> -through <object> -to <object> -name <string>
#path_delay -from <object> -through <object> -to <object> -delay <delay in ps> -name <string>
puts "Define Timing Constraints"
#suspend
define_clock -name CLKFS -period 31250 [vfind -port clk_fs]
define_clock -name CLKD1 -period 62500 [vfind -port clk_fso2]
define_clock -name CLKD2 -period 125000 [vfind -port clk_fso4]
path_disable -from [vfind -port rstb] 

puts "The number of exceptions is [llength [vfind "design:$DESIGN" -exception *]]"

if {![file exists ${_OUTPUTS_PATH}]} {
  file mkdir ${_OUTPUTS_PATH}
  puts "Creating directory ${_OUTPUTS_PATH}"
}

if {![file exists ${_REPORTS_PATH}]} {
  file mkdir ${_REPORTS_PATH}
  puts "Creating directory ${_REPORTS_PATH}"
}

if {![file exists ${_LOG_PATH}]} {
  file mkdir ${_LOG_PATH}
  puts "Creating directory ${_LOG_PATH}"
}

#### To turn off sequential merging on the design 
#### uncomment & use the following attributes.
##set_db / .optimize_merge_flops false 
##set_db / .optimize_merge_latches false 
#### For a particular instance use attribute 'optimize_merge_seqs' to turn off sequential merging. 



####################################################################################################
## Synthesizing to generic 
####################################################################################################

set_db / .syn_generic_effort $GEN_EFF
syn_generic
puts "Runtime & Memory after 'syn_generic'"
time_info GENERIC
report_dp > $_REPORTS_PATH/generic/${DESIGN}_datapath.rpt
write_snapshot -outdir $_REPORTS_PATH -tag generic
report_summary -directory $_REPORTS_PATH





####################################################################################################
## Synthesizing to gates
####################################################################################################


set_db / .syn_map_effort $MAP_OPT_EFF
syn_map
puts "Runtime & Memory after 'syn_map'"
time_info MAPPED
write_snapshot -outdir $_REPORTS_PATH -tag map
report_summary -directory $_REPORTS_PATH
report_dp > $_REPORTS_PATH/map/${DESIGN}_datapath.rpt



write_do_lec -revised_design fv_map -logfile ${_LOG_PATH}/rtl2intermediate.lec.log > ${_OUTPUTS_PATH}/rtl2intermediate.lec.do

## ungroup -threshold <value>

#######################################################################################################
## Optimize Netlist
#######################################################################################################

## Uncomment to remove assigns & insert tiehilo cells during Incremental synthesis
##set_db / .remove_assigns true 
##set_remove_assign_options -buffer_or_inverter <libcell> -design <design|subdesign> 
##set_db / .use_tiehilo_for_const <none|duplicate|unique> 
set_db / .syn_opt_effort $MAP_OPT_EFF
syn_opt
write_snapshot -outdir $_REPORTS_PATH -tag syn_opt
report_summary -directory $_REPORTS_PATH

puts "Runtime & Memory after 'syn_opt'"
time_info OPT




write_snapshot -outdir $_REPORTS_PATH -tag final
report_summary -directory $_REPORTS_PATH
## write_hdl  > ${_OUTPUTS_PATH}/${DESIGN}_m.v
## write_script > ${_OUTPUTS_PATH}/${DESIGN}_m.script
write_sdc > ${_OUTPUTS_PATH}/${DESIGN}_m.sdc


#################################
### write_do_lec
#################################


write_do_lec -golden_design fv_map -revised_design ${_OUTPUTS_PATH}/${DESIGN}_m.v -logfile  ${_LOG_PATH}/intermediate2final.lec.log > ${_OUTPUTS_PATH}/intermediate2final.lec.do
##Uncomment if the RTL is to be compared with the final netlist..
##write_do_lec -revised_design ${_OUTPUTS_PATH}/${DESIGN}_m.v -logfile ${_LOG_PATH}/rtl2final.lec.log > ${_OUTPUTS_PATH}/rtl2final.lec.do

puts "Final Runtime & Memory."
time_info FINAL
puts "============================"
puts "Synthesis Finished ........."
puts "============================"

file copy [get_db / .stdout_log] ${_LOG_PATH}/.

##quit
