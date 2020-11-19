/*
  Name: Shifter
  Copyright: 
  Author: Cumhur Erdin
  Date: 23/7/2018
  Description: Shifter from IF to BB
*/
module shifter
   (
    input	 [11:0]	i_in,
    input	 [11:0]	q_in,
	input 		rst_neg,
	input		clk,
	input		bypass,	
    output	reg	[11:0]	i_out,
	output	reg	[11:0]	q_out
    );

	reg		[2:0]	freq_45;
	reg		[12:0]	sin_out;
	reg		[12:0]	cos_out;

	reg		[23:0]	i_24;
	reg		[23:0]	q_24;

	always @(posedge clk or negedge rst_neg) 
	begin
    	if(!rst_neg)
			begin
				freq_45 = 0;
				i_out   = 0;
				q_out	= 0;
				sin_out = 0;
				cos_out = 0;
			end
		else if (bypass)
			begin
				i_out = i_in;
				q_out = q_in;
			end
    	else
			begin
				case (freq_45)
				0 : begin
					 sin_out = 0 ;
					 cos_out = 4095 ;
					end
				1 : begin
					 sin_out = 2896 ;
					 cos_out = 2896 ;
					end
				2 : begin
					 sin_out = 4095 ;
					 cos_out = 0 ;
					end
				3 : begin
					 sin_out = 2896 ;
					 cos_out = -2896 ;
					end
				4 : begin
					 sin_out = 0 ;
					 cos_out = -4095 ;
					end
				5 : begin
					 sin_out = -2896 ;
					 cos_out = -2896 ;
					end
				6 : begin
					 sin_out = -4095 ;
					 cos_out = 0 ;
					end
				7 : begin
					 sin_out = -2896 ;
					 cos_out = 2896 ;
					end
				endcase
				i_24  =  (($signed(i_in)*$signed(cos_out)) + ($signed(q_in)*$signed(sin_out)));
				i_out = i_24>>>12;
				q_24  =  (-($signed(i_in)*$signed(sin_out)) + ($signed(q_in)*$signed(cos_out)));
				q_out = q_24>>>12;
				freq_45 = freq_45 + 1;
			end
	end

endmodule
