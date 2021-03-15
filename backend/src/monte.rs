#![allow(dead_code)]
use super::board::*;

struct Node {
    visited_count: u32,
    win_count: u32,
    loss_count: u32,
    state: Board,
    untried_moves: Vec<Move>,
    children: Vec<Node>,
    player: u8,
}

impl Node {
    pub fn new(state: Board, player: u8) -> Self {
        Self {
            visited_count: 0,
            win_count: 0,
            loss_count: 0,
            state: state.clone(),
            untried_moves: state.gen_possible_moves(),
            children: vec![],
            player: player,
        }
    }

    pub fn expand(&mut self) {
        let mv = self.untried_moves.first().unwrap().clone();
        self.untried_moves.remove(0);
        let mut board = self.state.clone();
        board.place(mv.x, mv.y, self.player);
    }

    pub fn n(&self) -> i32 {
        self.visited_count as i32
    }

    pub fn q(&self) -> i32 {
        self.win_count as i32 - self.loss_count as i32
    }

    pub fn is_terminal_node(&self) -> bool {
        self.state.any_winner() == None
    }

    pub fn backpropagete(&mut self) {}
}
