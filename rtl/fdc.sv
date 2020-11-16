// This is an implementation of frequency down converter (fdc)
// Using cordic accumulator and fir filter

`define COMBINATORIAL
`include "cordic.v"

module fdc
#(
    parameter BW=12,
    parameter ABW=16,

)
(
    input clk,
    input rstb,
    input [BW-1:0] Iin, 
    input [BW-1:0] Qin, 
    input [BW-1:0] A,
    output [BW-1:0] Iout, 
    output [BW-1:0] Qout 
);
    reg [BW-1:0] IinR,QinR;
    reg [ABW-1:0] Phase;
    reg             clk2,clk4;
    wire [ABW-1:0] PhaseOut;

    wire [BW-1:0] Irot, Qrot;
    wire [BW-1:0] Idec1, Qdec1;
    wire [BW-1:0] Idec2, Qdec2;
    wire [BW-1:0] If1, Qf1;
    wire [BW-1:0] If2, Qf2;

    //Latch Inputs
    always_ff @(posedge clk or negedge rstb)
        if (rstb == 0)
         begin
            IinR <= 'd0;
            QinR <= 'd0;
         end   
        else
         begin
            IinR <= Iin;
            QinR <= Qin;
         end

    // Divide Clocks
    always_ff @(posedge clk or negedge rstb)
        if (rstb==0)
            clk2 <= 0;
        else
            clk2 <= ~clk2;

    alsways_ff @(posedge clk2 or negedge rstb)
        if (rstb==0)
            clk4 <= 0;
        else
            clk4 <= clk4;

    //Calculate Phase
    always_ff @(posedge clk or negedge rstb)
        if (rstb == 0)
            Phase <= 'd0;
        else
            Phase <= $signed(Phase) + $signed(A);

    //Gross Derotation
    TODO

    //Fine Derotation
    TODO

    //Filter 1
    fir #(.BW(12),.N(5)) uFIRI1 (
        .CK(clk),
        .RB(rstb),
        .X(I_rot),
        .C({12'd211,12'd420,12'd503,12'd420,12'd211}),
        .Y(I_f1));
    fir #(.BW(12),.N(5)) uFIRQ1 (
        .CK(clk),
        .RB(rstb),
        .X(Q_rot),
        .C({12'd211,12'd420,12'd503,12'd420,12'd211}),
        .Y(Q_f1));

    //Decimate 1

    always_ff @(posedge clk2 or negedge rstb)
        if (rstb==0)
         begin
            I_dec1 <= 'd0;
            Q_dec1 <= 'd0;
         end
        else
         begin
            I_dec1 <= I_f1;
            Q_dec1 <= Q_f1;
         end

    // Filter 2

    fir #(.BW(12),.N(5)) uFIRI2 (
        .CK(clk),
        .RB(rstb),
        .X(I_dec1),
        .C({12'd211,12'd420,12'd503,12'd420,12'd211}),
        .Y(I_f2));
    fir #(.BW(12),.N(5)) uFIRI2 (
        .CK(clk),
        .RB(rstb),
        .X(Q_dec1),
        .C({12'd211,12'd420,12'd503,12'd420,12'd211}),
        .Y(Q_f2));

    // Decimate 2
    //
    always_ff @(posedge clk4 or negedge rstb)
        if (rstb==0)
         begin
            I_dec2 <= 'd0;
            Q_dec2 <= 'd0;
         end
        else
         begin
            I_dec2 <= I_f2;
            Q_dec2 <= Q_f2;
         end
    
    // Connect to outputs

    assign Qout = Q_dec2;
    assign Iout = I_dec2;


endmodule
