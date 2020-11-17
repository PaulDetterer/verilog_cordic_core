
// Angle representation is always a 0.bbbbbb of a circle
// 100000 -> half circle (pi)
// 111111 -> 2*pi (almost)


module FDC
#(
    parameter BW = 12,
    parameter ABW = 10,
)
(
    input clk_fs,  //  Sampling clock
    input clk_fso2,  // Decimation clock fsample/2
    input clk_fso4,  // Decimation clock fsample/4
    input rstb, // Reset

    input [BW-1:0] Iin,
    input [BW-1:0] Qin,
    input [ABW-1:0] Wif,
    output [BW-1:0] Iout,
    output [BW-1:0] Qout
)
    reg  [BW-1:0] Iin_s, Qin_s;
    wire  [BW-1:0] Igr, Qgr;

    // Sample
    always_ff @(posedge clk_fs or negedge rstb)
        if (rstb == 0)
         begin
            Iin_s <= 'd0;
            Qin_s <= 'd0;
         end
        else
         begin
            Iin_s <= Iin;
            Qin_s <= Qin;
         end

    
    // Compute rotation angle theta 
    reg [ABW-1:0] theta; 

    always_ff @(posedge clk_fs or negedge rstb)
        if (rstb==0)
            theta <= 0;
        else
            theta <= unsigned(theta) + unsigned(Wif);

    // Gross Rotation
    always_comb
        case(theta[ABW-1:ABW-2])
            2'b00    :  {Igr,Qgr} = {Iin_s,Qin_s}; // rot 0
            2'b01    :  {Igr,Qgr} = {Qin_s,-signed(Iin_s)}; //rot p/2
            2'b10    :  {Igr,Qgr} = {-signed(Iin_s),-signed(Qin_s)}; //rot p
            2'b11    :  {Igr,Qgr} = {-signed(Qin_s),Iin_s}; // rot 3pi/2
        endcase

    // Fine Rotation
    // Define the angle function
    function [ABW-1:0] tangle;
        input [3:0] i;
        begin
            case(i)
			 3'd0 : tangle = 10'd128;
			 3'd1 : tangle = 10'd76;
			 3'd2 : tangle = 10'd40;
			 3'd3 : tangle = 10'd20;
			 3'd4 : tangle = 10'd10;
			 3'd5 : tangle = 10'd5;
			 3'd6 : tangle = 10'd3;
			 3'd7 : tangle = 10'd1;

            endcase
        end
    endfunction

    wire signed [BW-1:0] x [8-1:0];
    wire signed [BW-1:0] y [8-1:0];
    wire signed [ABW-1:0] z [8-1:0];

    assign x[0] = Igr;
    assign y[0] = Qgr;
    assign z[0] = theta;
    wire Ifrt = x[8-1];
    wire Qfrt = y[8-1];
    wire ThetaR = z[8-1];

    genvar i;
    generate for(i=0;i<8-1;i=i+1) begin
      derotator uDerotator (clk_fs,!rstb,x[i],y[i],z[i],x[i+1],y[i+1],z[i+1]);
      defparam uDerotator.BW = BW;
      defparam UDerotator.ABW = ABW;
      defparam uDerotator.iteration = i;
      defparam uDerotator.tangle = tanangle(i);
    end 
    endgenerate

    
    // Filter 1
    wire [BW-1:0] Ilpf1,Qlpf1;
    reg  [BW-1:0] Idec1,Qdec1;

    fir #(.BW(BW),.N(5),.S(BW-1)) uIFIR1 (
        .CK(clk_fs),
        .RB(rstb),
        .X(Ifrt),
        .C({12'd211,12'd420,12'd503,12'd420,12'd211}),
        .Y(Ilpf1));
    
    fir #(.BW(BW),.N(5),.S(BW-1)) uQFIR1 (
        .CK(clk_fs),
        .RB(rstb),
        .X(Qfrt),
        .C({12'd211,12'd420,12'd503,12'd420,12'd211}),
        .Y(Qlpf1));

    //Decimate 1

    always_ff @(posedge clk_fso2 or negedge rstb)
        if (rstb==0)
         begin
            Idec1 <= 'd0;
            Qdec1 <= 'd0;
         end
        else
         begin
            Idec1 <= Ilpf1;
            Qdec1 <= Qlpf1;
         end

    //Filter 2

    
    wire [BW-1:0] Ilpf2,Qlpf2;
    reg  [BW-1:0] Idec2,Qdec2;

    fir #(.BW(BW),.N(5),.S(BW-1)) uIFIR2 (
        .CK(clk_fso2),
        .RB(rstb),
        .X(Idec1),
        .C({12'd211,12'd420,12'd503,12'd420,12'd211}),
        .Y(Illpf2));
    
    fir #(.BW(BW),.N(5),.S(BW-1)) uQFIR2 (
        .CK(clk_fs),
        .RB(rstb),
        .X(Qdec1),
        .C({12'd211,12'd420,12'd503,12'd420,12'd211}),
        .Y(Qlpf2));
    
    //Decimate 2


    always_ff @(posedge clk_fso4 or negedge rstb)
        if (rstb==0)
         begin
            Idec2 <= 'd0;
            Qdec2 <= 'd0;
         end
        else
         begin
            Idec2 <= Ilpf2;
            Qdec2 <= Qlpf2;
         end

    assign Qout = Qdec2;
    assign Iout = Idec2;

endmodule    

