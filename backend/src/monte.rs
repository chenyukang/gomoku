#![allow(dead_code)]
use super::board::*;
use super::utils::*;
use std::cell::RefCell;
use std::rc::Rc;
use std::rc::Weak;

#[derive(Debug, Clone)]
struct Node {
    visited_count: u32,
    win_count: u32,
    loss_count: u32,
    state: Board,
    untried_moves: Vec<Move>,
    children: Vec<Rc<Node>>,
    player: u8,
    parent: Weak<RefCell<Node>>,
}

impl Node {
    pub fn new(state: Board, player: u8, parent: Weak<RefCell<Self>>) -> Self {
        Self {
            visited_count: 0,
            win_count: 0,
            loss_count: 0,
            state: state.clone(),
            untried_moves: state.gen_possible_moves(),
            children: vec![],
            player: player,
            parent: parent,
        }
    }

    pub fn expand(&mut self) -> Node {
        let mv = self.untried_moves.first().unwrap().clone();
        self.untried_moves.remove(0);
        let mut board = self.state.clone();
        board.place(mv.x, mv.y, self.player);
        let cell = RefCell::new(self.to_owned());
        let parent = Rc::downgrade(&Rc::new(cell.to_owned()));
        assert_eq!(parent.upgrade().is_some(), true);
        let child = Node::new(board, self.player, parent);
        self.children.push(Rc::new(child.to_owned()));
        child
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

    pub fn backpropagete(&mut self, winner: Option<u8>) {
        println!("now check ...");
        self.visited_count += 1;
        if winner == Some(self.player) {
            self.win_count += 1;
        } else if winner == Some(cfg::opponent(self.player)) {
            self.loss_count += 1;
        }
        if self.parent.upgrade().is_some() {
            let p = self.parent.upgrade().unwrap();
            p.borrow_mut().backpropagete(winner);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node_create() {
        let mut board = Board::new_default();
        board.place(5, 5, 1);
        let mut root = Node::new(board, 1, Weak::default());
        assert_eq!(root.parent.upgrade().is_none(), true);

        assert_eq!(root.untried_moves.len(), 8);

        let mut node = root.expand();
        assert_eq!(root.parent.upgrade().is_none(), true);
        assert_eq!(root.untried_moves.len(), 7);
        assert_eq!(node.parent.upgrade().is_none(), false);
        node.backpropagete(Some(2));
    }
}
