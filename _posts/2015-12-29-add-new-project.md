---
layout: post
title: "Adding a new project to the website" 
description: 
modified: 2015-12-29
tags: 
categories: project
---

This page will walk you through how to add a new project to TheCannon.io website. If you are ninja enough, you can also make these changes through a terminal by committing a new post to the repository rather than using the GitHub interface.


1. The website is stored in the `gh-pages` branch in the project's git repository. Visit <a href="https://github.com/andycasey/AnniesLasso/tree/gh-pages/_posts">the `gh-pages` branch on GitHub</a>.

2. Click the 'New file' button in the top right-hand side of that page.

3. Name your file according to the following convention: `YYYY-MM-DD-my-project-idea.md`.

4. Describe your project below using <a href="https://help.github.com/articles/github-flavored-markdown/">GitHub flavored Markdown</a>. You must have the `layout`, `title`, and `categories` information in the header as shown:
{% highlight text %}
---
layout: post
title: "My awesome project idea"
categories: project
---

We should do *ALL THE THINGS*.
{% endhighlight %}


Commit your new file to the repository using the form at the bottom of the GitHub page.

That's it! Your project should be listed on TheCannon.io in a few minutes. If it's not, <a href="mailto:arc@ast.cam.ac.uk">fire Andy</a>. 


