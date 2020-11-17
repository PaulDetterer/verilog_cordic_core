`timescale 1ns/1ps
module tb_FDC();
	integer File;
	integer synch=0;
	parameter Ps=31.25ns; // Nanoseconds
	parameter Ps2 = Ps * 2;
	parameter Ps4 = Ps * 4;
	parameter BW = 12;
	parameter ABW = 10;

	reg clk_fs, clk_fso2, clkfso4; // Clocks
	reg rstb // Reset Bar
	reg [BW-1:0] Iin, Qin; // Input Stimuli
	reg [ABW-1:0] Wif;
	wire [BW-1:0] Iout, Qout; // Output 

	clocking ckIn @(posedge clk_fs);
		output Iin, Qin, Wif;
	endclocking

	clocking ckOut @(posedge clk_fs04);
		input Iout,Qout;
	endclocking

	// Clock generation
	initial begin
		clk_fs <= 0;
		while(true)
			#(Ps/2.0) clk_fs <= ~clk_fs; 
	end
	
	initial begin
		clk_fso2 <= 0;
		while(true)
			@(posedge clk_fs) clk_fso2 <= ~clk_fso2; 
	end


	initial begin
		clk_fso4 <= 0;
		while(true)
			@(posedge clk_fso2) clk_fso4 <= ~clk_fso4; 
	end

	// Stimuli generation
	initial begin
		File = $fopen("stimuli/iq.txt","r");

		rstb <= 0;
		ckIn.Iin <= 'd0;
		ckIn.Qin <= 'd0;
		ckIn.Wif <= 'd0;
		for (int i=0; i<20; i=i+1)
			@(posedge clk_fso4);
		clkIn.Wif <= 'd38; 	
		rstb <= 1;
		while($eof(File)==0)
			@(posedge clk_fs) $fscanf(File,"%d\t%d\n",clkIn.Iin,clkIn.Qin);
		$fclose(File);
	end
endmodule
