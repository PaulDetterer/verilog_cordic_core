`timescale 1ns/1ps
module tb_FDC();
	integer File;
	integer synch=0;
	parameter Ps=31.25; // Nanoseconds
	parameter Ps2 = Ps * 2;
	parameter Ps4 = Ps * 4;
	parameter BW = 12;
	parameter ABW = 10;

	reg clk_fs, clk_fso2, clk_fso4; // Clocks
	reg rstb; // Reset Bar
	reg [BW-1:0] If, Qf; // Stimuli from file
	reg [BW-1:0] Iin, Qin; // Input Stimuli
	reg [ABW-1:0] Wif;
	wire [BW-1:0] Iout, Qout; // Output 

	clocking ckIn @(posedge clk_fs);
		output Iin, Qin, Wif;
	endclocking

	clocking ckOut @(posedge clk_fso4);
		input Iout,Qout;
	endclocking

	// Clock generation
	initial begin
		clk_fs <= 0;
		while(1)
			#(Ps/2.0) clk_fs <= ~clk_fs; 
	end
	
	initial begin
		clk_fso2 <= 0;
		while(1)
			@(posedge clk_fs) clk_fso2 <= ~clk_fso2; 
	end


	initial begin
		clk_fso4 <= 0;
		while(1)
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
		ckIn.Wif <= 'd64; 	

        $fscanf(File,"%d\t%d\n",If,Qf);
        $fscanf(File,"%d\t%d\n",If,Qf); // For sync
        ckIn.Iin <= If;
        ckIn.Qin <= Qf;
		rstb <= 1;
//        @(posedge clk_fs); // For Sync
		while($feof(File)==0)
         begin
			@(posedge clk_fs) $fscanf(File,"%d\t%d\n",If,Qf);
            ckIn.Iin <= If;
            ckIn.Qin <= Qf;
         end
		$fclose(File);
        $finish;
	end

    // Instanciate The DUT
    //
    MULFDC #(.ABW(ABW),.BW(BW)) u_DUT(
        .clk_fs(clk_fs),
        .clk_fso2(clk_fso2),
        .clk_fso4(clk_fso4),
        .rstb(rstb),
        .Iin(Iin),
        .Qin(Qin),
        .Wif(Wif),
        .Iout(Iout),
        .Qout(Qout)
    );
endmodule
