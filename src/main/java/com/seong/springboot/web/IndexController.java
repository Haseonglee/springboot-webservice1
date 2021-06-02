package com.seong.springboot.web;

import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;
//@Controller
@Controller
public class IndexController {

    @GetMapping("/index")
    public String index() {
        return "index";
    }
}